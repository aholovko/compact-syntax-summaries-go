package exclude

import (
	"go/parser"
	"go/token"
	"reflect"
	"testing"
)

func detectForTest(t *testing.T, code string) []string {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return Detect(f, []byte(code))
}

func TestDetectsGenerics(t *testing.T) {
	got := detectForTest(t, "package p\nfunc Map[T any](xs []T) []T { return xs }")
	if !reflect.DeepEqual(got, []string{"generics"}) {
		t.Fatalf("got %v, want [generics]", got)
	}
}

func TestDetectsCgoAndEmbed(t *testing.T) {
	code := "package p\n\nimport \"C\"\nimport _ \"embed\"\n\n//go:embed x.txt\nvar x string"
	got := detectForTest(t, code)
	want := []string{"cgo", "go:embed"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestDetectsBuildTagsAndUnsafe(t *testing.T) {
	code := "//go:build linux\n\npackage p\n\nimport \"unsafe\"\n\nfunc f(p unsafe.Pointer) uintptr { return uintptr(p) }"
	got := detectForTest(t, code)
	want := []string{"build_tags", "unsafe_ptr"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestEmbedLookAlikeNotFlagged(t *testing.T) {
	got := detectForTest(t, "package p\n//go:embedded note\nfunc f() {}")
	if len(got) != 0 {
		t.Fatalf("//go:embedded must not flag go:embed; got %v", got)
	}
}

func TestBuildLookAlikeNotFlagged(t *testing.T) {
	got := detectForTest(t, "//go:buildx linux\n\npackage p\nfunc f() {}")
	if len(got) != 0 {
		t.Fatalf("//go:buildx must not flag build_tags; got %v", got)
	}
}

func TestCleanFileHasNoExclusions(t *testing.T) {
	got := detectForTest(t, "package p\nfunc f() {}")
	if len(got) != 0 {
		t.Fatalf("got %v, want []", got)
	}
}

func TestBuildTagOnlyCountedInLeadingGroup(t *testing.T) {
	got := detectForTest(t, "package p\n\n//go:build linux\nfunc f() {}")
	if len(got) != 0 {
		t.Fatalf("non-leading //go:build must not flag build_tags; got %v", got)
	}
}
