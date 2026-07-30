package summary

import (
	"strings"
	"testing"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/parse"
	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/record"
)

func emitWithAnnotationsForTest(t *testing.T, code string) []record.Line {
	t.Helper()
	r, err := parse.Parse(code)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	e := newEmitter(r)
	e.walkRoots(r.Roots)
	e.emitTypeAnnotations(r.Roots)
	return e.lines
}

func TestParamAnnotationTier2(t *testing.T) {
	lines := emitWithAnnotationsForTest(t, "package p\nfunc f(items []Item) error { return nil }")
	if !containsText(lines, "types/annotations: items []Item", 2, 0) {
		t.Fatalf("missing param annotation; got tier2 %v", textsAtTier(lines, 2))
	}
}

func TestOneLinePerSymbol(t *testing.T) {
	lines := emitWithAnnotationsForTest(t, "package p\nfunc f(a int, b string) {}")
	got := textsAtTier(lines, 2)
	if len(got) != 2 || got[0] != "types/annotations: a int" || got[1] != "types/annotations: b string" {
		t.Fatalf("want one line per symbol, got %v", got)
	}
}

func TestGroupedNamesOneLinePerSymbol(t *testing.T) {
	lines := emitWithAnnotationsForTest(t, "package p\nfunc f(a, b int) {}")
	got := textsAtTier(lines, 2)
	want := []string{"types/annotations: a int", "types/annotations: b int"}
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("grouped names must be one line per symbol; got %v", got)
	}
}

func TestLocalTypedVarConstAreAnnotated(t *testing.T) {
	lines := emitWithAnnotationsForTest(t, "package p\nfunc f() { var n int; const k byte = 1 }")
	if !containsText(lines, "types/annotations: n int", 2, 0) || !containsText(lines, "types/annotations: k byte", 2, 0) {
		t.Fatalf("typed local var/const should be annotated; got tier2 %v", textsAtTier(lines, 2))
	}
}

func TestReceiverAndTypeParamsNotAnnotatedButParamIs(t *testing.T) {
	lines := emitWithAnnotationsForTest(t, "package p\nfunc (s *Server) Handle(w int) error { return nil }")
	got := textsAtTier(lines, 2)
	for _, l := range got {
		if strings.Contains(l, "s *Server") {
			t.Fatalf("method receiver must NOT be annotated; got %v", got)
		}
	}
	found := false
	for _, l := range got {
		if l == "types/annotations: w int" {
			found = true
		}
	}
	if !found {
		t.Fatalf("method param must be annotated; got %v", got)
	}
}

func TestGenericTypeParamsNotReSurfaced(t *testing.T) {
	lines := emitWithAnnotationsForTest(t, "package p\nfunc Map[T any](xs []T) []T { return xs }")
	got := textsAtTier(lines, 2)
	for _, l := range got {
		if strings.Contains(l, "T any") {
			t.Fatalf("generic type-params must not be re-surfaced at tier 2; got %v", got)
		}
	}
	found := false
	for _, l := range got {
		if l == "types/annotations: xs []T" {
			found = true
		}
	}
	if !found {
		t.Fatalf("generic func param must still be annotated; got %v", got)
	}
}
