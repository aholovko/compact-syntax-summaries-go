package parse

import (
	"strings"
	"testing"
)

func TestParseSelectsStrategy(t *testing.T) {
	cases := []struct {
		name string
		code string
		want Strategy
	}{
		{"file", "package p\nfunc f() {}", StrategyFile},
		{"package_decl", "func f() int { return 1 }", StrategyPackage},
		{"func_body_stmt", "x := 1\n_ = x", StrategyFuncBody},
		{"bare_expr_is_func_body", "a + b*2", StrategyFuncBody},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r, err := Parse(c.code)
			if err != nil {
				t.Fatalf("Parse(%q): %v", c.code, err)
			}
			if r.Strategy != c.want {
				t.Fatalf("Parse(%q) strategy = %q, want %q", c.code, r.Strategy, c.want)
			}
			if len(r.Roots) == 0 {
				t.Fatalf("Parse(%q) returned no roots", c.code)
			}
		})
	}
}

func TestParseFailureSurfacesDiagnostic(t *testing.T) {
	_, err := Parse("func (")
	if err == nil {
		t.Fatal("expected a parse error")
	}
	msg := err.Error()
	if !strings.Contains(msg, "no parse strategy accepted the snippet") {
		t.Fatalf("missing generic message: %q", msg)
	}
	if !strings.Contains(msg, "file:") {
		t.Fatalf("expected a file-strategy diagnostic; got %q", msg)
	}
	_, err2 := Parse("func (")
	if err2.Error() != msg {
		t.Fatalf("non-deterministic parse error:\n %q\n %q", msg, err2.Error())
	}
}

func TestTryExprDirectly(t *testing.T) {
	r, ok := tryExpr("a + b*2")
	if !ok || r.Strategy != StrategyExpr {
		t.Fatalf("tryExpr should accept a bare expression: ok=%v strategy=%q", ok, r.Strategy)
	}
	if len(r.Roots) != 1 {
		t.Fatalf("expr should yield one root, got %d", len(r.Roots))
	}
}

func TestParseRootsExcludeWrapper(t *testing.T) {
	r, err := Parse("x := 1\n_ = x")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if r.Strategy != StrategyFuncBody {
		t.Fatalf("strategy = %q", r.Strategy)
	}
	if len(r.Roots) != 2 {
		t.Fatalf("want 2 inner statements as roots, got %d", len(r.Roots))
	}
}
