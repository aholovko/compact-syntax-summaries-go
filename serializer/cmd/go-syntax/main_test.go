package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type summaryRecord struct {
	ID         string  `json:"id"`
	OK         bool    `json:"ok"`
	Lines      []line  `json:"lines"`
	ParseError *string `json:"parse_error"`
}

type line struct {
	Text string `json:"text"`
}

func TestSummarizePreservesOrderAndIDsSkipsBlankLinesAndHandlesLargeRecords(t *testing.T) {
	largeCode := "package p\n// " + strings.Repeat("x", 17<<20) + "\nvar Y = 1"
	input := strings.Join([]string{
		inputRecord(t, "first", "package p"),
		"",
		inputRecord(t, "malformed", "!!!not go"),
		" \t",
		inputRecord(t, "large", largeCode),
		"",
	}, "\n")

	got := summarizeFromStdin(t, input)
	gotIDs := make([]string, len(got))
	for i, record := range got {
		gotIDs[i] = record.ID
	}
	wantIDs := []string{"first", "malformed", "large"}
	if !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("output IDs = %v, want %v", gotIDs, wantIDs)
	}
	if !got[2].OK {
		t.Fatalf("large record was not summarized successfully: %+v", got[2])
	}
}

func TestSummarizeValidFunctionEmitsRealSummary(t *testing.T) {
	got := summarizeFromStdin(t, inputRecord(t, "valid", "package p\nfunc f() error { return nil }")+"\n")
	if len(got) != 1 {
		t.Fatalf("got %d records, want 1", len(got))
	}
	if got[0].ID != "valid" || !got[0].OK {
		t.Fatalf("valid function result = %+v, want id valid and ok true", got[0])
	}
	for _, line := range got[0].Lines {
		if line.Text == "return: nil" {
			return
		}
	}
	t.Fatalf("valid function summary has no return line: %+v", got[0].Lines)
}

func TestSummarizeMalformedGoEmitsStructuredFailure(t *testing.T) {
	got := summarizeFromStdin(t, inputRecord(t, "broken", "!!!not go")+"\n")
	if len(got) != 1 {
		t.Fatalf("got %d records, want 1", len(got))
	}
	if got[0].ID != "broken" || got[0].OK {
		t.Fatalf("malformed result = %+v, want id broken and ok false", got[0])
	}
	if got[0].ParseError == nil || strings.TrimSpace(*got[0].ParseError) == "" {
		t.Fatalf("malformed result has no parse_error: %+v", got[0])
	}
}

func TestSummarizeProcessesFinalRecordWithoutTrailingNewline(t *testing.T) {
	got := summarizeFromStdin(t, inputRecord(t, "final", "package p"))
	if len(got) != 1 || got[0].ID != "final" || !got[0].OK {
		t.Fatalf("unterminated final record result = %+v, want one successful final record", got)
	}
}

func TestRunCommandRejectsMissingAndUnknownCommands(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		wantInError string
	}{
		{name: "missing", args: nil, wantInError: "missing"},
		{name: "unknown", args: []string{"unknown"}, wantInError: "unknown"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := runCommand(test.args, strings.NewReader(""), &bytes.Buffer{})
			if err == nil {
				t.Fatal("runCommand returned nil, want an error")
			}
			if !strings.Contains(strings.ToLower(err.Error()), test.wantInError) {
				t.Fatalf("runCommand error = %q, want it to contain %q", err, test.wantInError)
			}
		})
	}
}

func TestRunCommandUsesNamedLocalInputAndOutput(t *testing.T) {
	dir := t.TempDir()
	inPath := filepath.Join(dir, "input.jsonl")
	outPath := filepath.Join(dir, "output.jsonl")
	input := []byte(inputRecord(t, "file", "package p\nfunc f() error { return nil }") + "\n")
	if err := os.WriteFile(inPath, input, 0o600); err != nil {
		t.Fatal(err)
	}

	err := runCommand(
		[]string{"summarize", "--in", inPath, "--out", outPath},
		forbiddenReader{},
		forbiddenWriter{},
	)
	if err != nil {
		t.Fatalf("runCommand with named files: %v", err)
	}
	output, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	got := decodeSummaries(t, output)
	if len(got) != 1 || got[0].ID != "file" || !got[0].OK {
		t.Fatalf("named output records = %+v, want one successful file record", got)
	}
}

func TestRunCommandPropagatesOutputStatErrorsBeforeCreation(t *testing.T) {
	dir := t.TempDir()
	inPath := filepath.Join(dir, "input.jsonl")
	outPath := filepath.Join(dir, "output.jsonl")
	input := []byte(inputRecord(t, "keep", "package p") + "\n")
	if err := os.WriteFile(inPath, input, 0o600); err != nil {
		t.Fatal(err)
	}

	originalStat := statFile
	wantErr := errors.New("injected output stat failure")
	statFile = func(path string) (os.FileInfo, error) {
		if path == outPath {
			return nil, wantErr
		}
		return originalStat(path)
	}
	t.Cleanup(func() { statFile = originalStat })

	err := runCommand(
		[]string{"summarize", "--in", inPath, "--out", outPath},
		strings.NewReader(""),
		&bytes.Buffer{},
	)
	if err != wantErr {
		t.Fatalf("runCommand error = %v, want original stat error %v", err, wantErr)
	}
	if _, err := os.Stat(outPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("output was created despite stat failure: %v", err)
	}
	got, err := os.ReadFile(inPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, input) {
		t.Fatalf("input changed from %q to %q", input, got)
	}
}

func TestRunCommandRejectsInputOutputAliasesWithoutChangingInput(t *testing.T) {
	for _, test := range []struct {
		name     string
		hardLink bool
	}{
		{name: "identical path"},
		{name: "hard-link alias", hardLink: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			inPath := filepath.Join(dir, "input.jsonl")
			original := []byte(inputRecord(t, "keep", "package p") + "\n")
			if err := os.WriteFile(inPath, original, 0o600); err != nil {
				t.Fatal(err)
			}

			outPath := inPath
			if test.hardLink {
				outPath = filepath.Join(dir, "output.jsonl")
				if err := os.Link(inPath, outPath); err != nil {
					t.Fatal(err)
				}
			}

			err := runCommand(
				[]string{"summarize", "--in", inPath, "--out", outPath},
				strings.NewReader(""),
				&bytes.Buffer{},
			)
			if err == nil {
				t.Fatal("runCommand returned nil, want an input/output alias error")
			}
			got, readErr := os.ReadFile(inPath)
			if readErr != nil {
				t.Fatal(readErr)
			}
			if !bytes.Equal(got, original) {
				t.Fatalf("input changed from %q to %q", original, got)
			}
		})
	}
}

type forbiddenReader struct{}

func (forbiddenReader) Read([]byte) (int, error) {
	return 0, errors.New("stdin must not be read when --in is set")
}

type forbiddenWriter struct{}

func (forbiddenWriter) Write([]byte) (int, error) {
	return 0, errors.New("stdout must not be written when --out is set")
}

func inputRecord(t *testing.T, id, code string) string {
	t.Helper()
	record, err := json.Marshal(map[string]string{"id": id, "code": code})
	if err != nil {
		t.Fatal(err)
	}
	return string(record)
}

func summarizeFromStdin(t *testing.T, input string) []summaryRecord {
	t.Helper()
	var output bytes.Buffer
	if err := runCommand([]string{"summarize"}, strings.NewReader(input), &output); err != nil {
		t.Fatalf("runCommand summarize: %v", err)
	}
	return decodeSummaries(t, output.Bytes())
}

func decodeSummaries(t *testing.T, output []byte) []summaryRecord {
	t.Helper()
	if len(output) == 0 {
		t.Fatal("summarize emitted no records")
	}
	output = bytes.TrimSuffix(output, []byte{'\n'})
	rawRecords := bytes.Split(output, []byte{'\n'})
	records := make([]summaryRecord, len(rawRecords))
	for i, raw := range rawRecords {
		if len(bytes.TrimSpace(raw)) == 0 {
			t.Fatalf("output record %d is blank", i)
		}
		if err := json.Unmarshal(raw, &records[i]); err != nil {
			t.Fatalf("decode output record %d: %v", i, err)
		}
	}
	return records
}
