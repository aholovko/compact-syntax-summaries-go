package summary

import (
	"testing"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/record"
)

func textsAtTier(lines []record.Line, tier int) []string {
	var out []string
	for _, l := range lines {
		if l.Tier == tier {
			out = append(out, l.Text)
		}
	}
	return out
}

func TestForLoopHeaderTier0(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(items []int) { for i := 0; i < len(items); i++ { _ = i } }")
	want := "loop: for index i; condition i < len(items); update i++"
	if !containsText(lines, want, 0, 1) {
		t.Fatalf("missing %q at tier0 depth1; got %v", want, textsAtTier(lines, 0))
	}
}

func TestForInitRendersAllIndexNames(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(n int) { for i, j := 0, n; i < j; i++ { _ = i } }")
	want := "loop: for index i, j; condition i < j; update i++"
	if !containsText(lines, want, 0, 1) {
		t.Fatalf("multi-var for-init must render all index names; got %v", textsAtTier(lines, 0))
	}
}

func TestIfHeaderRidesPredicate(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(x int) { if x > 0 { _ = x } }")
	if !containsText(lines, "conditional: x > 0", 0, 1) {
		t.Fatalf("missing conditional; got %v", textsAtTier(lines, 0))
	}
}

func TestElseIfChain(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(x int) { if x > 0 { _ = x } else if x < 0 { _ = x } else { _ = x } }")
	for _, want := range []string{"conditional: x > 0", "conditional: else x < 0", "conditional: else"} {
		if !containsText(lines, want, 0, 1) {
			t.Fatalf("missing %q at tier0 depth1; got %v", want, textsAtTier(lines, 0))
		}
	}
}

func TestReturnHeaderRidesExpr(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f() error { return nil }")
	if !containsText(lines, "return: nil", 0, 1) {
		t.Fatalf("missing return; got %v", textsAtTier(lines, 0))
	}
}

func TestSwitchAndCase(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(x int) {\nswitch x {\ncase 1:\n_ = x\ndefault:\n_ = x\n}\n}")
	if !containsText(lines, "switch: x", 0, 1) {
		t.Fatalf("missing switch; got %v", textsAtTier(lines, 0))
	}
	if !containsText(lines, "case: 1", 0, 2) {
		t.Fatalf("missing case; got %v", textsAtTier(lines, 0))
	}
}

func TestRangeLoop(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(m map[string]int) { for k, v := range m { _, _ = k, v } }")
	if !containsText(lines, "loop: range key k; value v; over m", 0, 1) {
		t.Fatalf("missing range; got %v", textsAtTier(lines, 0))
	}
}

func TestDeferGoSelect(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(ch chan int) { defer close(ch); go work(); select { case <-ch: } }")
	if !containsText(lines, "defer: close(ch)", 0, 1) {
		t.Fatalf("missing defer; got %v", textsAtTier(lines, 0))
	}
	if !containsText(lines, "go: work()", 0, 1) {
		t.Fatalf("missing go; got %v", textsAtTier(lines, 0))
	}
	if !containsText(lines, "select:", 0, 1) {
		t.Fatalf("missing select; got %v", textsAtTier(lines, 0))
	}
}

func TestIfInitRidesOnHeader(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f() { if err := g(); err != nil { h() } }")
	if !containsText(lines, "conditional: err := g(); err != nil", 0, 1) {
		t.Fatalf("if-init missing; got %v", textsAtTier(lines, 0))
	}
}

func TestElseIfInitRidesOnHeader(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(x int) { if x > 0 { h() } else if err := g(); err != nil { h() } }")
	if !containsText(lines, "conditional: else err := g(); err != nil", 0, 1) {
		t.Fatalf("else-if-init missing; got %v", textsAtTier(lines, 0))
	}
}

func TestSwitchInitRidesOnHeader(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f() {\nswitch x := tag(); x {\ncase 1:\n_ = x\n}\n}")
	if !containsText(lines, "switch: x := tag(); x", 0, 1) {
		t.Fatalf("switch-init missing; got %v", textsAtTier(lines, 0))
	}
}

func TestLabeledLoopNotDropped(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(items []int) {\nLoop:\nfor i := 0; i < len(items); i++ {\nbreak Loop\n}\n}")
	if !containsText(lines, "loop: for index i; condition i < len(items); update i++", 0, 1) {
		t.Fatalf("labeled loop dropped; got %v", textsAtTier(lines, 0))
	}
}

func TestCaseBodyTrailingCommentAtBodyDepth(t *testing.T) {
	code := "package p\nfunc f(x int) {\nswitch x {\ncase 1:\n\tfoo()\n\t// trailing\ncase 2:\n\tbar()\n}\n}"
	lines := emitWithCommentsForTest(t, code)
	if !containsText(lines, "comment: line", 1, 3) {
		t.Fatalf("trailing case-body comment not at body depth 3; comment depths=%v", commentLineDepths(lines))
	}
}

func commentLineDepths(lines []record.Line) []int {
	var out []int
	for _, l := range lines {
		if l.Text == "comment: line" {
			out = append(out, l.Depth)
		}
	}
	return out
}

func containsText(lines []record.Line, text string, tier, depth int) bool {
	for _, l := range lines {
		if l.Text == text && l.Tier == tier && l.Depth == depth {
			return true
		}
	}
	return false
}
