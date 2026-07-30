package summary

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/parse"
)

func TestAssignmentTier1(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f(items []Item) { items[0].Status = \"pending\" }")
	if !containsText(lines, "assignment: items[0].Status = \"pending\"", 1, 1) {
		t.Fatalf("missing assignment; got tier1 %v", textsAtTier(lines, 1))
	}
}

func TestCallTier1(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f() { doWork(1, 2) }")
	if !containsText(lines, "call: doWork(1, 2)", 1, 1) {
		t.Fatalf("missing call; got tier1 %v", textsAtTier(lines, 1))
	}
}

func TestNonDuplicationPredicateNotReEmitted(t *testing.T) {
	code := "package p\nfunc f(items []Item) { for i := 0; i < len(items); i++ { if items[i].Status == \"\" { items[i].Status = \"pending\" } } }"
	lines := emitForTest(t, code)
	for _, l := range lines {
		if l.Tier == 1 && l.Text == "access: items[i].Status" {
			t.Fatalf("predicate access was wrongly re-emitted as tier-1")
		}
	}
	if !containsText(lines, "assignment: items[i].Status = \"pending\"", 1, 3) {
		t.Fatalf("assignment missing; got tier1 %v", textsAtTier(lines, 1))
	}
}

func TestCompositeLiteralTier1(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f() { Point{X: 1, Y: 2} }")
	if !containsText(lines, "composite-literal: Point{X: 1, Y: 2}", 1, 1) {
		t.Fatalf("missing composite literal; got tier1 %v", textsAtTier(lines, 1))
	}
}

func TestSubExpressionsRideOnStatementLine(t *testing.T) {
	lines := emitForTest(t, "package p\nfunc f() { p := Point{X: doWork(a[i])} }")
	got := textsAtTier(lines, 1)
	if len(got) != 1 || got[0] != "assignment: p := Point{X: doWork(a[i])}" {
		t.Fatalf("sub-expressions must ride on the assignment, not get own lines; got tier1 %v", got)
	}
}

func TestParenthesizedExprStatementsUnwrap(t *testing.T) {
	for _, tc := range []struct{ code, want string }{
		{"package p\nfunc f() { (doWork(1, 2)) }", "call: doWork(1, 2)"},
		{"package p\nfunc f(m map[string]int) { (m[\"k\"]) }", "access: m[\"k\"]"},
		{"package p\nfunc f(ch chan int) { (<-ch) }", "channel: receive <-ch"},
		{"package p\nfunc f() { ((doWork())) }", "call: doWork()"},
	} {
		lines := emitForTest(t, tc.code)
		if !containsText(lines, tc.want, 1, 1) {
			t.Fatalf("paren stmt %q: missing %q; got tier1 %v", tc.code, tc.want, textsAtTier(lines, 1))
		}
	}
}

func TestExprStrategyEmission(t *testing.T) {
	fset := token.NewFileSet()
	e, err := parser.ParseExprFrom(fset, "", "doWork(a, b)", 0)
	if err != nil {
		t.Fatalf("parse expr: %v", err)
	}
	r := parse.Result{Strategy: parse.StrategyExpr, Roots: []ast.Node{e}, Src: []byte("doWork(a, b)"), Fset: fset}
	lines := Emit(r)
	if len(lines) != 1 || lines[0].Tier != 1 || lines[0].Text != "call: doWork(a, b)" {
		t.Fatalf("expr emission: want one tier-1 'call: doWork(a, b)', got %v", lines)
	}
}
