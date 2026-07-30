package record

import (
	"encoding/json"
	"testing"
)

func TestSegmentMarshalsToSingleKeyObject(t *testing.T) {
	got, err := json.Marshal(Segment{Kind: "a", Text: "func "})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(got) != `{"a":"func "}` {
		t.Fatalf("got %s, want {\"a\":\"func \"}", got)
	}
}

func TestSegmentRoundTrips(t *testing.T) {
	var s Segment
	if err := json.Unmarshal([]byte(`{"v":"items []Item"}`), &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if s.Kind != "v" || s.Text != "items []Item" {
		t.Fatalf("got %+v", s)
	}
}

func TestOutputEmitsEmptyArraysNotNull(t *testing.T) {
	got, err := json.Marshal(Output{ID: "x", OK: false})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"id":"x","ok":false,"parse_strategy":null,"type_facts_available":false,"lines":[],"excluded_constructs":[],"parse_error":null}`
	if string(got) != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}
