package summary

import "go/ast"

// Type annotations include only named syntax that is safe to expose verbatim.
// Receivers, generic parameter lists, unnamed results, and interface method names stay out of scope.
func (e *emitter) emitTypeAnnotations(roots []ast.Node) {
	for _, root := range roots {
		ast.Inspect(root, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.FuncType:
				e.emitFieldList(x.Params)
				e.emitFieldList(x.Results)
			case *ast.StructType:
				e.emitFieldList(x.Fields)
			case *ast.ValueSpec:
				if x.Type != nil {
					e.emitField(&ast.Field{Names: x.Names, Type: x.Type})
				}
			}
			return true
		})
	}
}

func (e *emitter) emitFieldList(fl *ast.FieldList) {
	if fl == nil {
		return
	}
	for _, f := range fl.List {
		e.emitField(f)
	}
}

func (e *emitter) emitField(f *ast.Field) {
	if f.Type == nil || len(f.Names) == 0 {
		return
	}
	for _, name := range f.Names {
		e.add(newLine(2, 0,
			authored("types/annotations: "),
			verbatim(e.src0(name)),
			authored(" "),
			verbatim(e.src0(f.Type)),
		))
	}
}
