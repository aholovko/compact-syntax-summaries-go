package summary

import "go/ast"

func (e *emitter) setupComments(file *ast.File) {
	if file == nil {
		return
	}
	e.docc = map[*ast.Comment]bool{}
	markDoc := func(g *ast.CommentGroup) {
		if g == nil {
			return
		}
		for _, c := range g.List {
			e.docc[c] = true
		}
	}
	ast.Inspect(file, func(n ast.Node) bool {
		switch d := n.(type) {
		case *ast.FuncDecl:
			markDoc(d.Doc)
		case *ast.GenDecl:
			markDoc(d.Doc)
		case *ast.TypeSpec:
			markDoc(d.Doc)
		case *ast.ValueSpec:
			markDoc(d.Doc)
		case *ast.Field:
			markDoc(d.Doc)
		}
		return true
	})
	for _, grp := range file.Comments {
		e.comments = append(e.comments, grp.List...)
	}
}

func (e *emitter) flushRest() {
	for e.ci < len(e.comments) {
		e.emitCommentMarker(e.comments[e.ci], 0)
		e.ci++
	}
}
