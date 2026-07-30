// Package record defines the go-syntax JSONL input and output schema.
package record

import (
	"encoding/json"
	"fmt"
)

// Input is the label-blind JSONL record accepted by go-syntax.
type Input struct {
	ID   string `json:"id"`
	Code string `json:"code"`
}

// Segment separates serializer-authored text from verbatim source text.
type Segment struct {
	Kind string
	Text string
}

// MarshalJSON encodes a segment as a single-key authored/verbatim object.
func (s Segment) MarshalJSON() ([]byte, error) {
	if s.Kind != "a" && s.Kind != "v" {
		return nil, fmt.Errorf("record: segment kind %q must be \"a\" or \"v\"", s.Kind)
	}
	return json.Marshal(map[string]string{s.Kind: s.Text})
}

// UnmarshalJSON decodes the single-key authored/verbatim segment object.
func (s *Segment) UnmarshalJSON(b []byte) error {
	var m map[string]string
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	if len(m) != 1 {
		return fmt.Errorf("record: segment must have exactly one key, got %d", len(m))
	}
	for k, v := range m {
		if k != "a" && k != "v" {
			return fmt.Errorf("record: segment key %q must be \"a\" or \"v\"", k)
		}
		s.Kind, s.Text = k, v
	}
	return nil
}

// Line stores rendered text with segment provenance for label-blindness checks.
type Line struct {
	Tier     int       `json:"tier"`
	Depth    int       `json:"depth"`
	Text     string    `json:"text"`
	Segments []Segment `json:"segments"`
}

// Output is one go-syntax JSONL result, successful or failed.
type Output struct {
	ID                 string   `json:"id"`
	OK                 bool     `json:"ok"`
	ParseStrategy      *string  `json:"parse_strategy"`
	TypeFactsAvailable bool     `json:"type_facts_available"`
	Lines              []Line   `json:"lines"`
	ExcludedConstructs []string `json:"excluded_constructs"`
	ParseError         *string  `json:"parse_error"`
}

// MarshalJSON encodes empty lines and excluded_constructs as [], not null.
func (o Output) MarshalJSON() ([]byte, error) {
	type alias Output
	a := alias(o)
	if a.Lines == nil {
		a.Lines = []Line{}
	}
	if a.ExcludedConstructs == nil {
		a.ExcludedConstructs = []string{}
	}
	return json.Marshal(a)
}
