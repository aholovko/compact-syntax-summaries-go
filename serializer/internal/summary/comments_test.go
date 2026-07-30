package summary

import (
	"strings"
	"testing"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/parse"
	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/record"
)

func emitWithCommentsForTest(t *testing.T, code string) []record.Line {
	t.Helper()
	r, err := parse.Parse(code)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	e := newEmitter(r)
	e.setupComments(r.File)
	e.walkRoots(r.Roots)
	e.flushRest()
	return e.lines
}

func TestCommentKinds(t *testing.T) {
	code := "package p\n// doc\nfunc f() {\n\t// line\n\t/* block */\n\t_ = 1\n}"
	lines := emitWithCommentsForTest(t, code)
	if !containsText(lines, "comment: doc", 1, 0) {
		t.Fatalf("doc marker missing at depth 0; got %v", textsAtTier(lines, 1))
	}
	if !containsText(lines, "comment: line", 1, 1) || !containsText(lines, "comment: block", 1, 1) {
		t.Fatalf("line/block markers missing at depth 1; got %v", textsAtTier(lines, 1))
	}
}

func TestCommentsInterleavedInSourceOrder(t *testing.T) {
	code := "package p\nfunc f() {\n\t// before\n\tx = 1\n\t// after\n\ty = 2\n}"
	lines := emitWithCommentsForTest(t, code)
	var seq []string
	for _, l := range lines {
		if l.Tier == 1 {
			seq = append(seq, l.Text)
		}
	}
	want := []string{"comment: line", "assignment: x = 1", "comment: line", "assignment: y = 2"}
	if len(seq) != len(want) {
		t.Fatalf("interleave: got %v, want %v", seq, want)
	}
	for i := range want {
		if seq[i] != want[i] {
			t.Fatalf("interleave order: got %v, want %v", seq, want)
		}
	}
}

func TestCommentCarriesNoText(t *testing.T) {
	lines := emitWithCommentsForTest(t, "package p\n// secret rule name\nfunc f() {}")
	for _, l := range lines {
		if l.Tier == 1 && len(l.Text) > len("comment: doc")+2 {
			t.Fatalf("comment marker leaked text: %q", l.Text)
		}
	}
}

func TestCommentInCaseBodyInterleaved(t *testing.T) {
	code := "package p\nfunc f(x int) {\nswitch x {\ncase 1:\n// note\n_ = x\n}\n}"
	lines := emitWithCommentsForTest(t, code)
	var seq []string
	for _, l := range lines {
		if l.Depth == 3 && l.Tier == 1 {
			seq = append(seq, l.Text)
		}
	}
	if len(seq) < 2 || seq[0] != "comment: line" || seq[1] != "assignment: _ = x" {
		t.Fatalf("case-body comment must precede the stmt at depth 3; got %v", seq)
	}
}

func TestInlineCommentStrippedFromVerbatim(t *testing.T) {
	code := "package p\nfunc f() {\n\tx := /* note */ 1\n}"
	lines := emitWithCommentsForTest(t, code)
	markers := 0
	for _, l := range lines {
		if strings.HasPrefix(l.Text, "comment:") {
			markers++
		}
		if strings.Contains(l.Text, "note") || strings.Contains(l.Text, "/*") {
			t.Fatalf("comment text leaked into line: %q", l.Text)
		}
	}
	if markers != 1 {
		t.Fatalf("want exactly one comment marker, got %d", markers)
	}
}

func TestHeaderCommentAtHeaderDepth(t *testing.T) {
	code := "package p\nfunc f(x int) {\n\tif /* why */ x > 0 {\n\t\tg()\n\t}\n}"
	lines := emitWithCommentsForTest(t, code)
	if !containsText(lines, "comment: block", 1, 1) {
		t.Fatalf("header comment not at header depth 1; tier-1 lines=%v", textsAtTier(lines, 1))
	}
}

func TestStructFieldCommentNotInTypeLine(t *testing.T) {
	code := "package p\ntype T struct {\n// inner\nA int\n}"
	lines := emitWithCommentsForTest(t, code)
	for _, l := range lines {
		if l.Tier == 0 && (strings.Contains(l.Text, "//") || strings.Contains(l.Text, "inner")) {
			t.Fatalf("comment text leaked into tier-0 line: %q", l.Text)
		}
	}
	foundHeader, foundMarker := false, false
	for _, l := range lines {
		if l.Tier == 0 && l.Text == "type T struct" {
			foundHeader = true
		}
		if l.Tier == 1 && (l.Text == "comment: line" || l.Text == "comment: doc") {
			foundMarker = true
		}
	}
	if !foundHeader {
		t.Fatalf("missing header-only type line; got %v", lines)
	}
	if !foundMarker {
		t.Fatalf("missing kind-only comment marker; got %v", lines)
	}
}
