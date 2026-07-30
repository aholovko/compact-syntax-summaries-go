package summary

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/allowlist"
)

func sourceWithoutComments(code string) string {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return normalizeWS(code) // parse-fail fixtures emit no verbatim
	}
	src := []byte(code)
	var out []byte
	cur := 0
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			lo := fset.Position(c.Pos()).Offset
			hi := fset.Position(c.End()).Offset
			if lo > cur {
				out = append(out, src[cur:lo]...)
			}
			if hi > cur {
				cur = hi
			}
		}
	}
	if cur < len(src) {
		out = append(out, src[cur:]...)
	}
	return normalizeWS(string(out))
}

func TestNoEmbeddedNewlinesInCorpus(t *testing.T) {
	matches, err := filepath.Glob("testdata/*.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no corpus")
	}
	for _, goPath := range matches {
		code, err := os.ReadFile(goPath)
		if err != nil {
			t.Fatal(err)
		}
		for _, l := range Summarize("id", string(code)).Lines {
			if strings.Contains(l.Text, "\n") {
				t.Fatalf("%s: Line.text has embedded newline: %q", goPath, l.Text)
			}
		}
	}
}

func TestLabelBlindnessAcrossCorpus(t *testing.T) {
	matches, err := filepath.Glob("testdata/*.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no corpus")
	}
	for _, goPath := range matches {
		code, err := os.ReadFile(goPath)
		if err != nil {
			t.Fatal(err)
		}
		srcNoComments := sourceWithoutComments(string(code))
		out := Summarize("id", string(code))
		for li, l := range out.Lines {
			var sb strings.Builder
			for _, s := range l.Segments {
				sb.WriteString(s.Text)
			}
			if sb.String() != l.Text {
				t.Fatalf("%s line %d: text != join(segments)", goPath, li)
			}
			for _, s := range l.Segments {
				switch s.Kind {
				case "a":
					if bad, ok := allowlist.Validate(allowlist.Tokenize(s.Text)); !ok {
						t.Fatalf("%s line %d: authored token %q not in allowlist", goPath, li, bad)
					}
				case "v":
					if !strings.Contains(srcNoComments, s.Text) {
						t.Fatalf("%s line %d: verbatim %q not in comment-stripped source", goPath, li, s.Text)
					}
				default:
					t.Fatalf("%s line %d: segment kind %q", goPath, li, s.Kind)
				}
			}
		}
	}
}
