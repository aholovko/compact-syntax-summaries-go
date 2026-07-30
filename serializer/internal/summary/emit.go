package summary

import (
	"go/ast"
	"go/token"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/parse"
	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/record"
)

type emitter struct {
	src      []byte
	fset     *token.FileSet
	lines    []record.Line
	emitted  map[ast.Node]bool
	comments []*ast.Comment
	ci       int
	docc     map[*ast.Comment]bool
}

func newEmitter(r parse.Result) *emitter {
	return &emitter{src: r.Src, fset: r.Fset, emitted: map[ast.Node]bool{}}
}

func (e *emitter) walkRoots(roots []ast.Node) {
	for _, root := range roots {
		e.flushComments(root.Pos(), 0)
		e.emitNode(root, 0)
	}
}

// Emit returns structural lines without comments or type annotations.
func Emit(r parse.Result) []record.Line {
	e := newEmitter(r)
	e.walkRoots(r.Roots)
	return e.lines
}

func (e *emitter) add(l record.Line) { e.lines = append(e.lines, l) }

func (e *emitter) src0(n ast.Node) string { return e.sliceNoComments(n.Pos(), n.End()) }

func (e *emitter) emitNode(n ast.Node, depth int) {
	switch x := n.(type) {
	case *ast.FuncDecl:
		e.add(e.funcSignature(x, depth))
		if x.Body != nil {
			e.emitBodyBlock(x.Body, depth)
		}
	case *ast.GenDecl:
		e.emitGenDecl(x, depth)
	case ast.Stmt:
		e.emitStmt(x, depth)
	case ast.Expr:
		e.emitExprDetail(x, depth)
	}
}

func (e *emitter) funcSignature(d *ast.FuncDecl, depth int) record.Line {
	segs := []record.Segment{authored("func ")}
	if d.Recv != nil && len(d.Recv.List) > 0 {
		segs = append(segs, verbatim(e.src0(d.Recv)), authored(" "))
	}
	segs = append(segs, verbatim(e.src0(d.Name)))
	sigStart := d.Type.Params.Pos()
	sigEnd := d.Type.Params.End()
	if d.Type.Results != nil {
		sigEnd = d.Type.Results.End()
	}
	segs = append(segs, verbatim(e.sliceRange(sigStart, sigEnd)))
	return newLine(0, depth, segs...)
}

func (e *emitter) sliceRange(lo, hi token.Pos) string { return e.sliceNoComments(lo, hi) }

func (e *emitter) sliceNoComments(lo, hi token.Pos) string {
	a := e.fset.Position(lo).Offset
	b := e.fset.Position(hi).Offset
	var out []byte
	cur := a
	for _, c := range e.comments {
		cLo := e.fset.Position(c.Pos()).Offset
		cHi := e.fset.Position(c.End()).Offset
		if cHi <= cur {
			continue
		}
		if cLo >= b {
			break
		}
		if cLo > cur {
			out = append(out, e.src[cur:cLo]...)
		}
		cur = cHi
	}
	if cur < b {
		out = append(out, e.src[cur:b]...)
	}
	return string(out)
}

func (e *emitter) emitGenDecl(d *ast.GenDecl, depth int) {
	for _, spec := range d.Specs {
		e.flushComments(spec.Pos(), depth)
		switch s := spec.(type) {
		case *ast.TypeSpec:
			conn := " "
			if s.Assign.IsValid() {
				conn = " = " // keep aliases distinct from definitions
			}
			switch s.Type.(type) {
			case *ast.StructType:
				// Fields are emitted later as annotations.
				e.add(newLine(0, depth, authored("type "), verbatim(e.src0(s.Name)), authored(conn+"struct")))
			case *ast.InterfaceType:
				e.add(newLine(0, depth, authored("type "), verbatim(e.src0(s.Name)), authored(conn+"interface")))
			default:
				// Separate slices omit any generic type-param list.
				e.add(newLine(0, depth, authored("type "), verbatim(e.src0(s.Name)), authored(conn), verbatim(e.src0(s.Type))))
			}
		case *ast.ValueSpec:
			kw := "var "
			if d.Tok == token.CONST {
				kw = "const "
			}
			e.add(newLine(0, depth, authored(kw), verbatim(e.src0(s))))
		}
	}
	e.flushComments(d.End(), depth)
}

func (e *emitter) emitBlock(b *ast.BlockStmt, depth int) {
	for _, s := range b.List {
		e.flushComments(s.Pos(), depth)
		e.emitStmt(s, depth)
	}
	e.flushComments(b.Rbrace, depth)
}

func (e *emitter) emitClauses(body *ast.BlockStmt, depth int) {
	for i, clause := range body.List {
		e.flushComments(clause.Pos(), depth)
		e.emitStmt(clause, depth)
		boundary := body.Rbrace
		if i+1 < len(body.List) {
			boundary = body.List[i+1].Pos()
		}
		e.flushComments(boundary, depth+1)
	}
}

func (e *emitter) emitBodyBlock(body *ast.BlockStmt, headerDepth int) {
	e.flushComments(body.Pos(), headerDepth)
	e.emitBlock(body, headerDepth+1)
}

func (e *emitter) emitBodyClauses(body *ast.BlockStmt, headerDepth int) {
	e.flushComments(body.Pos(), headerDepth)
	e.emitClauses(body, headerDepth+1)
}

func (e *emitter) flushComments(before token.Pos, depth int) {
	for e.ci < len(e.comments) && e.comments[e.ci].Pos() < before {
		e.emitCommentMarker(e.comments[e.ci], depth)
		e.ci++
	}
}

func (e *emitter) emitCommentMarker(c *ast.Comment, depth int) {
	kind := "line"
	if e.docc[c] {
		kind = "doc"
	} else if len(c.Text) >= 2 && c.Text[1] == '*' {
		kind = "block"
	}
	e.add(newLine(1, depth, authored("comment: "), authored(kind)))
}

func (e *emitter) emitStmt(s ast.Stmt, depth int) {
	switch st := s.(type) {
	case *ast.ForStmt:
		e.add(e.forHeader(st, depth))
		if st.Body != nil {
			e.emitBodyBlock(st.Body, depth)
		}
	case *ast.RangeStmt:
		e.add(e.rangeHeader(st, depth))
		if st.Body != nil {
			e.emitBodyBlock(st.Body, depth)
		}
	case *ast.IfStmt:
		segs := []record.Segment{authored("conditional: ")}
		if st.Init != nil {
			segs = append(segs, e.markVerbatim(st.Init), authored("; "))
		}
		segs = append(segs, e.condSeg(st.Cond))
		e.add(newLine(0, depth, segs...))
		if st.Body != nil {
			e.emitBodyBlock(st.Body, depth)
		}
		e.emitElse(st.Else, depth)
	case *ast.BlockStmt:
		e.emitBlock(st, depth+1)
	case *ast.SwitchStmt:
		segs := []record.Segment{authored("switch:")}
		if st.Init != nil {
			segs = append(segs, authored(" "), e.markVerbatim(st.Init), authored(";"))
		}
		if st.Tag != nil {
			segs = append(segs, authored(" "), e.markVerbatim(st.Tag))
		}
		e.add(newLine(0, depth, segs...))
		if st.Body != nil {
			e.emitBodyClauses(st.Body, depth)
		}
	case *ast.TypeSwitchStmt:
		segs := []record.Segment{authored("switch:")}
		if st.Init != nil {
			segs = append(segs, authored(" "), e.markVerbatim(st.Init), authored(";"))
		}
		segs = append(segs, authored(" "), e.markVerbatim(st.Assign))
		e.add(newLine(0, depth, segs...))
		if st.Body != nil {
			e.emitBodyClauses(st.Body, depth)
		}
	case *ast.CaseClause:
		if len(st.List) == 0 {
			e.add(newLine(0, depth, authored("default:")))
		} else {
			e.add(newLine(0, depth, authored("case: "), e.exprListSeg(st.List)))
		}
		e.flushComments(st.Colon, depth)
		for _, inner := range st.Body {
			e.flushComments(inner.Pos(), depth+1)
			e.emitStmt(inner, depth+1)
		}
	case *ast.SelectStmt:
		e.add(newLine(0, depth, authored("select:")))
		if st.Body != nil {
			e.emitBodyClauses(st.Body, depth)
		}
	case *ast.CommClause:
		if st.Comm != nil {
			e.add(newLine(0, depth, authored("case: "), verbatim(e.src0(st.Comm))))
		} else {
			e.add(newLine(0, depth, authored("default:")))
		}
		e.flushComments(st.Colon, depth)
		for _, inner := range st.Body {
			e.flushComments(inner.Pos(), depth+1)
			e.emitStmt(inner, depth+1)
		}
	case *ast.DeferStmt:
		e.add(newLine(0, depth, authored("defer: "), verbatim(e.src0(st.Call))))
	case *ast.GoStmt:
		e.add(newLine(0, depth, authored("go: "), verbatim(e.src0(st.Call))))
	case *ast.ReturnStmt:
		if len(st.Results) == 0 {
			e.add(newLine(0, depth, authored("return:")))
		} else {
			e.add(newLine(0, depth, authored("return: "), e.exprListSeg(st.Results)))
		}
	case *ast.LabeledStmt:
		// Labels are ignored; the labeled statement still matters.
		e.emitStmt(st.Stmt, depth)
	case *ast.BranchStmt:
		// Branch statements are a v1 omission.
	default:
		e.emitDetail(s, depth)
	}
}

func (e *emitter) forHeader(st *ast.ForStmt, depth int) record.Line {
	segs := []record.Segment{authored("loop: for")}
	if st.Init != nil {
		if as, ok := st.Init.(*ast.AssignStmt); ok && len(as.Lhs) > 0 {
			names := e.sliceRange(as.Lhs[0].Pos(), as.Lhs[len(as.Lhs)-1].End())
			segs = append(segs, authored(" index "), verbatim(names))
		}
	}
	if st.Cond != nil {
		segs = append(segs, authored("; condition "), e.condSeg(st.Cond))
	}
	if st.Post != nil {
		segs = append(segs, authored("; update "), e.markVerbatim(st.Post))
	}
	return newLine(0, depth, segs...)
}

func (e *emitter) rangeHeader(st *ast.RangeStmt, depth int) record.Line {
	segs := []record.Segment{authored("loop: range")}
	if st.Key != nil {
		segs = append(segs, authored(" key "), e.markVerbatim(st.Key))
	}
	if st.Value != nil {
		segs = append(segs, authored("; value "), e.markVerbatim(st.Value))
	}
	segs = append(segs, authored("; over "), e.markVerbatim(st.X))
	return newLine(0, depth, segs...)
}

func unparen(x ast.Expr) ast.Expr {
	for {
		p, ok := x.(*ast.ParenExpr)
		if !ok {
			return x
		}
		x = p.X
	}
}

func (e *emitter) condSeg(x ast.Expr) record.Segment {
	e.emitted[x] = true
	return verbatim(e.src0(x))
}

func (e *emitter) markVerbatim(n ast.Node) record.Segment {
	e.emitted[n] = true
	return verbatim(e.src0(n))
}

func (e *emitter) exprListSeg(list []ast.Expr) record.Segment {
	for _, x := range list {
		e.emitted[x] = true
	}
	return verbatim(e.sliceRange(list[0].Pos(), list[len(list)-1].End()))
}

func (e *emitter) emitElse(els ast.Stmt, depth int) {
	switch x := els.(type) {
	case nil:
		return
	case *ast.IfStmt:
		segs := []record.Segment{authored("conditional: else ")}
		if x.Init != nil {
			segs = append(segs, e.markVerbatim(x.Init), authored("; "))
		}
		segs = append(segs, e.condSeg(x.Cond))
		e.add(newLine(0, depth, segs...))
		if x.Body != nil {
			e.emitBodyBlock(x.Body, depth)
		}
		e.emitElse(x.Else, depth)
	case *ast.BlockStmt:
		e.add(newLine(0, depth, authored("conditional: else")))
		e.emitBodyBlock(x, depth)
	}
}

func (e *emitter) emitDetail(s ast.Stmt, depth int) {
	switch st := s.(type) {
	case *ast.AssignStmt:
		if e.emitted[st] {
			return
		}
		e.add(newLine(1, depth, authored("assignment: "), verbatim(e.src0(st))))
	case *ast.ExprStmt:
		e.emitExprDetail(st.X, depth)
	case *ast.SendStmt:
		e.add(newLine(1, depth, authored("channel: send "), verbatim(e.src0(st))))
	case *ast.DeclStmt:
		if gd, ok := st.Decl.(*ast.GenDecl); ok {
			e.emitGenDecl(gd, depth)
		}
	case *ast.IncDecStmt:
		if e.emitted[st] {
			return
		}
		e.add(newLine(1, depth, authored("assignment: "), verbatim(e.src0(st))))
	}
}

func (e *emitter) emitExprDetail(x ast.Expr, depth int) {
	x = unparen(x)
	if e.emitted[x] {
		return
	}
	switch ex := x.(type) {
	case *ast.CallExpr:
		e.add(newLine(1, depth, authored("call: "), verbatim(e.src0(ex))))
	case *ast.CompositeLit:
		e.add(newLine(1, depth, authored("composite-literal: "), verbatim(e.src0(ex))))
	case *ast.UnaryExpr:
		if ex.Op == token.ARROW {
			e.add(newLine(1, depth, authored("channel: receive "), verbatim(e.src0(ex))))
			return
		}
		e.add(newLine(1, depth, authored("access: "), verbatim(e.src0(ex))))
	case *ast.SelectorExpr, *ast.IndexExpr:
		e.add(newLine(1, depth, authored("access: "), verbatim(e.src0(ex))))
	}
}
