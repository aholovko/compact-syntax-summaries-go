package allowlist

import "testing"

func TestValidateAcceptsAuthoredRoles(t *testing.T) {
	if bad, ok := Validate([]string{"loop:", "for", "index", "condition", "update"}); !ok {
		t.Fatalf("expected valid, rejected %q", bad)
	}
}

func TestValidateRejectsUnknownToken(t *testing.T) {
	if _, ok := Validate([]string{"loop:", "redundant"}); ok {
		t.Fatal("expected 'redundant' to be rejected")
	}
}

func TestTokenizeSplitsOnSeparators(t *testing.T) {
	got := Tokenize("loop: for index ")
	want := []string{"loop:", "for", "index"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}
