// Package exclude detects Go constructs outside serializer v1 scope.
package exclude

import (
	"go/ast"
	"sort"
	"strings"
)

// Detect returns stable output names for constructs outside v1 scope:
//   - generics:   TypeParams on FuncDecl/TypeSpec
//   - go:embed:   //go:embed directive comment
//   - cgo:        import "C"
//   - build_tags: //go:build (or legacy // +build) in the leading comments
//   - unsafe_ptr: use of unsafe.Pointer / uintptr
func Detect(file *ast.File, src []byte) []string {
	found := map[string]bool{}

	if file != nil {
		ast.Inspect(file, func(n ast.Node) bool {
			switch d := n.(type) {
			case *ast.FuncDecl:
				if d.Type != nil && d.Type.TypeParams != nil && len(d.Type.TypeParams.List) > 0 {
					found["generics"] = true
				}
			case *ast.TypeSpec:
				if d.TypeParams != nil && len(d.TypeParams.List) > 0 {
					found["generics"] = true
				}
			case *ast.ImportSpec:
				if d.Path != nil && d.Path.Value == `"C"` {
					found["cgo"] = true
				}
			case *ast.SelectorExpr:
				if id, ok := d.X.(*ast.Ident); ok && id.Name == "unsafe" && d.Sel.Name == "Pointer" {
					found["unsafe_ptr"] = true
				}
			case *ast.Ident:
				// v1 is syntactic-only, so any uintptr identifier conservatively flags unsafe_ptr.
				if d.Name == "uintptr" {
					found["unsafe_ptr"] = true
				}
			}
			return true
		})
		for _, grp := range file.Comments {
			for _, c := range grp.List {
				if isDirective(c.Text, "//go:embed") {
					found["go:embed"] = true
				}
				// Go honors build tags only before the package clause, so later look-alikes must not affect
				// exclusion counts.
				if c.Pos() < file.Package && isBuildConstraint(c.Text) {
					found["build_tags"] = true
				}
			}
		}
	} else {
		for _, line := range leadingDirectiveLines(src) {
			t := strings.TrimSpace(line)
			if isDirective(t, "//go:embed") {
				found["go:embed"] = true
			}
			if isBuildConstraint(t) {
				found["build_tags"] = true
			}
		}
	}

	out := make([]string, 0, len(found))
	for k := range found {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func isBuildConstraint(text string) bool {
	return isDirective(text, "//go:build") || isDirective(text, "// +build")
}

func isDirective(text, directive string) bool {
	if !strings.HasPrefix(text, directive) {
		return false
	}
	rest := text[len(directive):]
	return rest == "" || rest[0] == ' ' || rest[0] == '\t'
}

func leadingDirectiveLines(src []byte) []string {
	var out []string
	for _, line := range strings.Split(string(src), "\n") {
		t := strings.TrimSpace(line)
		if t == "" || strings.HasPrefix(t, "//") || strings.HasPrefix(t, "/*") {
			out = append(out, line)
			continue
		}
		break
	}
	return out
}
