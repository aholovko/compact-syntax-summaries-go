package summary

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

var updateGolden = flag.Bool("update", false, "regenerate golden .json files instead of comparing")

func TestGoldenCorpus(t *testing.T) {
	matches, err := filepath.Glob("testdata/*.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no golden snippets found")
	}
	for _, goPath := range matches {
		goPath := goPath
		t.Run(filepath.Base(goPath), func(t *testing.T) {
			code, err := os.ReadFile(goPath)
			if err != nil {
				t.Fatal(err)
			}
			jsonPath := goPath[:len(goPath)-3] + ".json"
			got := Summarize("sha256:golden", string(code))
			gotBytes, err := json.MarshalIndent(got, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if *updateGolden {
				if err := os.WriteFile(jsonPath, append(gotBytes, '\n'), 0o644); err != nil {
					t.Fatal(err)
				}
				t.Logf("updated %s", jsonPath)
				return
			}
			wantBytes, err := os.ReadFile(jsonPath)
			if err != nil {
				t.Fatalf("missing golden json (run with -update): %v", err)
			}
			if !bytes.Equal(gotBytes, bytes.TrimRight(wantBytes, "\n")) {
				t.Fatalf("golden mismatch for %s:\n got: %s\nwant: %s", goPath, gotBytes, wantBytes)
			}
		})
	}
}

func TestDeterministicRunTwice(t *testing.T) {
	matches, err := filepath.Glob("testdata/*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, goPath := range matches {
		code, err := os.ReadFile(goPath)
		if err != nil {
			t.Fatal(err)
		}
		a, err := json.Marshal(Summarize("id", string(code)))
		if err != nil {
			t.Fatal(err)
		}
		b, err := json.Marshal(Summarize("id", string(code)))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a, b) {
			t.Fatalf("non-deterministic output for %s", goPath)
		}
	}
}
