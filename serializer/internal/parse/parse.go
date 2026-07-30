// Package parse turns Go snippets into AST roots.
package parse

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
)

// Strategy identifies how Parse accepted a snippet.
type Strategy string

const (
	// StrategyFile means the snippet parsed as a complete Go file.
	StrategyFile Strategy = "file"
	// StrategyPackage means Parse added a package clause before parsing.
	StrategyPackage Strategy = "package"
	// StrategyFuncBody means Parse wrapped the snippet in a synthetic function body.
	StrategyFuncBody Strategy = "func_body"
	// StrategyExpr means the snippet parsed as a standalone expression.
	StrategyExpr Strategy = "expr"
)

// Result is the AST and source state produced by the accepted parse strategy.
type Result struct {
	Strategy Strategy
	Roots    []ast.Node
	Src      []byte
	Fset     *token.FileSet
	File     *ast.File
}

// Parse tries supported snippet shapes in order and returns the first complete AST that contains no parser
// recovery nodes.
func Parse(code string) (Result, error) {
	if r, ok := tryFile(code); ok {
		return r, nil
	}
	if r, ok := tryPackage(code); ok {
		return r, nil
	}
	if r, ok := tryFuncBody(code); ok {
		return r, nil
	}
	if r, ok := tryExpr(code); ok {
		return r, nil
	}
	return Result{}, fmt.Errorf("no parse strategy accepted the snippet (%s)", fileDiagnostic(code))
}

func fileDiagnostic(code string) string {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments|parser.SkipObjectResolution)
	switch {
	case err != nil:
		return "file: " + err.Error()
	case hasBadNodes(f):
		return "file: parser produced error-recovery (Bad*) nodes"
	default:
		return "file: parsed, but a later strategy was selected and also rejected"
	}
}

func tryFile(code string) (Result, bool) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil || f == nil || hasBadNodes(f) {
		return Result{}, false
	}
	return Result{
		Strategy: StrategyFile,
		Roots:    declNodes(f.Decls),
		Src:      []byte(code),
		Fset:     fset,
		File:     f,
	}, true
}

func tryPackage(code string) (Result, bool) {
	const prefix = "package p\n"
	src := prefix + code
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil || f == nil || hasBadNodes(f) {
		return Result{}, false
	}
	return Result{
		Strategy: StrategyPackage,
		Roots:    declNodes(f.Decls),
		Src:      []byte(src),
		Fset:     fset,
		File:     f,
	}, true
}

func tryFuncBody(code string) (Result, bool) {
	const prefix = "package p\nfunc _() {\n"
	src := prefix + code + "\n}"
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil || f == nil || hasBadNodes(f) {
		return Result{}, false
	}
	if len(f.Decls) != 1 {
		return Result{}, false
	}
	fn, ok := f.Decls[0].(*ast.FuncDecl)
	if !ok || fn.Body == nil {
		return Result{}, false
	}
	roots := make([]ast.Node, 0, len(fn.Body.List))
	for _, s := range fn.Body.List {
		roots = append(roots, s)
	}
	return Result{
		Strategy: StrategyFuncBody,
		Roots:    roots,
		Src:      []byte(src),
		Fset:     fset,
		File:     f,
	}, true
}

func tryExpr(code string) (Result, bool) {
	fset := token.NewFileSet()
	e, err := parser.ParseExprFrom(fset, "", code, parser.SkipObjectResolution)
	if err != nil || e == nil || hasBadNodes(e) {
		return Result{}, false
	}
	return Result{
		Strategy: StrategyExpr,
		Roots:    []ast.Node{e},
		Src:      []byte(code),
		Fset:     fset,
	}, true
}

func declNodes(decls []ast.Decl) []ast.Node {
	out := make([]ast.Node, 0, len(decls))
	for _, d := range decls {
		out = append(out, d)
	}
	return out
}

func hasBadNodes(n ast.Node) bool {
	bad := false
	ast.Inspect(n, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.BadDecl, *ast.BadStmt, *ast.BadExpr:
			bad = true
			return false
		}
		return !bad
	})
	return bad
}
