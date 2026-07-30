package summary

import (
	"go/ast"
	"testing"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/parse"
)

func TestVerbatimSliceStripsComments(t *testing.T) {
	for _, tc := range []struct{ code, want string }{
		{"package p\nvar x = a + b", "a + b"},
		{"package p\nvar x = a + /* c */ b", "a + b"},
	} {
		r, err := parse.Parse(tc.code)
		if err != nil {
			t.Fatalf("parse %q: %v", tc.code, err)
		}
		e := newEmitter(r)
		e.setupComments(r.File)
		got := ""
		for _, root := range r.Roots {
			ast.Inspect(root, func(n ast.Node) bool {
				if be, ok := n.(*ast.BinaryExpr); ok {
					got = normalizeWS(e.src0(be))
					return false
				}
				return true
			})
		}
		if got != tc.want {
			t.Fatalf("src0(%q) = %q, want %q", tc.code, got, tc.want)
		}
	}
}
