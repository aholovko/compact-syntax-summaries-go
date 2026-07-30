package summary

import (
	"strings"
	"testing"
)

var canonicalStmtTypes = []string{
	"BadStmt", "DeclStmt", "EmptyStmt", "LabeledStmt", "ExprStmt", "SendStmt",
	"IncDecStmt", "AssignStmt", "GoStmt", "DeferStmt", "ReturnStmt", "BranchStmt",
	"BlockStmt", "IfStmt", "CaseClause", "SwitchStmt", "TypeSwitchStmt",
	"CommClause", "SelectStmt", "ForStmt", "RangeStmt",
}

type stmtCov struct {
	name    string
	snippet string
	handled bool
	reason  string
}

var stmtCoverage = []stmtCov{
	{name: "DeclStmt", snippet: "var x int", handled: true},
	{name: "EmptyStmt", snippet: ";", handled: false, reason: "no structural content"},
	{name: "LabeledStmt", snippet: "L: return", handled: true},
	{name: "ExprStmt", snippet: "f()", handled: true},
	{name: "SendStmt", snippet: "ch <- 1", handled: true},
	{name: "IncDecStmt", snippet: "i++", handled: true},
	{name: "AssignStmt", snippet: "x = 1", handled: true},
	{name: "GoStmt", snippet: "go f()", handled: true},
	{name: "DeferStmt", snippet: "defer f()", handled: true},
	{name: "ReturnStmt", snippet: "return", handled: true},
	{name: "BranchStmt", snippet: "break", handled: false, reason: "break/continue/goto/fallthrough open no scope and carry no studied-check signal"},
	{name: "BlockStmt", snippet: "{ return }", handled: true},
	{name: "IfStmt", snippet: "if x > 0 { f() }", handled: true},
	{name: "SwitchStmt", snippet: "switch x { }", handled: true},
	{name: "CaseClause", snippet: "switch x { case 1: f() }", handled: true},
	{name: "TypeSwitchStmt", snippet: "switch x.(type) { }", handled: true},
	{name: "SelectStmt", snippet: "select { }", handled: true},
	{name: "CommClause", snippet: "select { case <-ch: f() }", handled: true},
	{name: "ForStmt", snippet: "for { }", handled: true},
	{name: "RangeStmt", snippet: "for range xs { }", handled: true},
	{name: "BadStmt", snippet: "", handled: false, reason: "parse-failure recovery node; never reached when ok=true"},
}

func structuralLineCount(t *testing.T, snippet string) int {
	t.Helper()
	code := "package p\nfunc _t() {\n" + snippet + "\n}"
	n := 0
	for _, l := range emitForTest(t, code) {
		if (l.Tier == 0 || l.Tier == 1) && !strings.HasPrefix(l.Text, "comment:") {
			n++
		}
	}
	return n
}

func TestStatementCoverageHandledOrIntentionallyOmitted(t *testing.T) {
	baseline := structuralLineCount(t, "")
	for _, c := range stmtCoverage {
		if c.snippet == "" {
			continue
		}
		got := structuralLineCount(t, c.snippet)
		if c.handled && got <= baseline {
			t.Errorf("%s classified handled but emitted no structural line for %q", c.name, c.snippet)
		}
		if !c.handled && got != baseline {
			t.Errorf("%s classified intentionally-omitted (%s) but emitted %d structural line(s) for %q", c.name, c.reason, got-baseline, c.snippet)
		}
	}

	ledger := map[string]bool{}
	for _, c := range stmtCoverage {
		if ledger[c.name] {
			t.Fatalf("duplicate ledger entry for %s", c.name)
		}
		ledger[c.name] = true
		if !c.handled && c.reason == "" {
			t.Errorf("%s is omitted but carries no reason; declare why it is out of scope", c.name)
		}
	}
	canonical := map[string]bool{}
	for _, n := range canonicalStmtTypes {
		canonical[n] = true
		if !ledger[n] {
			t.Errorf("ast.%s is not in the coverage ledger; classify it handled or intentionally-omitted", n)
		}
	}
	for n := range ledger {
		if !canonical[n] {
			t.Errorf("ledger lists %q which is not a known ast.Stmt type", n)
		}
	}
}

func TestBranchStatementsAreOmitted(t *testing.T) {
	for _, snippet := range []string{
		"for { break }",
		"for { continue }",
		"switch x { case 1: fallthrough; case 2: }",
		"L: for { goto L }",
	} {
		lines := emitForTest(t, "package p\nfunc _t(x int) {\n"+snippet+"\n}")
		for _, l := range lines {
			if strings.Contains(l.Text, "break") || strings.Contains(l.Text, "continue") ||
				strings.Contains(l.Text, "fallthrough") || strings.Contains(l.Text, "goto") {
				t.Fatalf("branch statement leaked into output for %q: %q", snippet, l.Text)
			}
		}
	}
}
