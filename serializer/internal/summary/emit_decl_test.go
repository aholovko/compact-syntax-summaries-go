package summary

import (
	"testing"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/parse"
	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/record"
)

func emitForTest(t *testing.T, code string) []record.Line {
	t.Helper()
	r, err := parse.Parse(code)
	if err != nil {
		t.Fatalf("parse %q: %v", code, err)
	}
	return Emit(r)
}

func firstLineText(lines []record.Line) string {
	if len(lines) == 0 {
		return ""
	}
	return lines[0].Text
}

func TestFuncSignatureTier0(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc ProcessItems(items []Item) error { return nil }")
	if got := firstLineText(lines); got != "func ProcessItems(items []Item) error" {
		t.Fatalf("func sig = %q", got)
	}
	if lines[0].Tier != 0 || lines[0].Depth != 0 {
		t.Fatalf("tier/depth = %d/%d", lines[0].Tier, lines[0].Depth)
	}
}

func TestMethodSignatureKeepsReceiver(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc (s *Server) Handle(w int) error { return nil }")
	if got := firstLineText(lines); got != "func (s *Server) Handle(w int) error" {
		t.Fatalf("method sig = %q", got)
	}
}

func TestGenericFuncElidesTypeParams(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc Map[T any](xs []T) []T { return xs }")
	if got := firstLineText(lines); got != "func Map(xs []T) []T" {
		t.Fatalf("generic sig = %q (type params must be elided)", got)
	}
}

func TestTypeAndConstDeclTier0(t *testing.T) {
	lines := emitForTest(t, "package p\ntype Status int\nconst Active Status = 1")
	if lines[0].Text != "type Status int" {
		t.Fatalf("type decl = %q", lines[0].Text)
	}
	if lines[1].Text != "const Active Status = 1" {
		t.Fatalf("const decl = %q", lines[1].Text)
	}
}

func TestStructDefHeader(t *testing.T) {
	lines := emitForTest(t, "package p\ntype Point struct{ X, Y int }")
	if !containsText(lines, "type Point struct", 0, 0) {
		t.Fatalf("struct header = %v", textsAtTier(lines, 0))
	}
}

func TestStructAliasShowsMarker(t *testing.T) {
	lines := emitForTest(t, "package p\ntype T = struct{ X int }")
	if !containsText(lines, "type T = struct", 0, 0) {
		t.Fatalf("struct-alias marker missing; got %v", textsAtTier(lines, 0))
	}
}

func TestInterfaceAliasShowsMarker(t *testing.T) {
	lines := emitForTest(t, "package p\ntype I = interface{ M() }")
	if !containsText(lines, "type I = interface", 0, 0) {
		t.Fatalf("interface-alias marker missing; got %v", textsAtTier(lines, 0))
	}
}

func TestStructHeaderNoCommentLeak(t *testing.T) {
	lines := emitForTest(t, "package p\ntype T struct /* secret */ {\n\tX int\n}")
	if !containsText(lines, "type T struct", 0, 0) {
		t.Fatalf("struct header leaked comment text; got %v", textsAtTier(lines, 0))
	}
}

func TestGroupedConstCommentsInterleaved(t *testing.T) {
	lines := emitWithCommentsForTest(t, "package p\nconst (\n\tA = 1 // ca\n\tB = 2 // cb\n)")
	idxA, idxB, firstComment := -1, -1, -1
	for i, l := range lines {
		switch l.Text {
		case "const A = 1":
			idxA = i
		case "const B = 2":
			idxB = i
		case "comment: line":
			if firstComment == -1 {
				firstComment = i
			}
		}
	}
	if idxA < 0 || idxB < 0 || idxA >= firstComment || firstComment >= idxB {
		t.Fatalf("expected a comment between const A and const B; idxA=%d firstComment=%d idxB=%d", idxA, firstComment, idxB)
	}
}
