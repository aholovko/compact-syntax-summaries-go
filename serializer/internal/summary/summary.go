package summary

import (
	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/exclude"
	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/parse"
	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/record"
)

// Summarize returns a syntax-only summary.
func Summarize(id, code string) record.Output {
	r, err := parse.Parse(code)
	if err != nil {
		msg := err.Error()
		return record.Output{
			ID:                 id,
			OK:                 false,
			ParseStrategy:      nil,
			TypeFactsAvailable: false,
			Lines:              []record.Line{},
			ExcludedConstructs: detectFailExclusions(code),
			ParseError:         &msg,
		}
	}
	e := newEmitter(r)
	e.setupComments(r.File)
	e.walkRoots(r.Roots)
	e.flushRest()
	e.emitTypeAnnotations(r.Roots)

	strategy := string(r.Strategy)
	return record.Output{
		ID:                 id,
		OK:                 true,
		ParseStrategy:      &strategy,
		TypeFactsAvailable: false,
		Lines:              e.lines,
		ExcludedConstructs: exclude.Detect(r.File, r.Src),
		ParseError:         nil,
	}
}

func detectFailExclusions(code string) []string {
	return exclude.Detect(nil, []byte(code))
}
