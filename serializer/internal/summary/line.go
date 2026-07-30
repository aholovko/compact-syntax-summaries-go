package summary

import (
	"strings"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/record"
)

func authored(text string) record.Segment { return record.Segment{Kind: "a", Text: text} }

func normalizeWS(s string) string { return strings.Join(strings.Fields(s), " ") }

func verbatim(text string) record.Segment { return record.Segment{Kind: "v", Text: normalizeWS(text)} }

func newLine(tier, depth int, segs ...record.Segment) record.Line {
	var sb strings.Builder
	for _, s := range segs {
		sb.WriteString(s.Text)
	}
	return record.Line{Tier: tier, Depth: depth, Text: sb.String(), Segments: segs}
}
