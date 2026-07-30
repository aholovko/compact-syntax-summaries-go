// Command go-syntax emits compact Go syntax summaries.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/record"
	"github.com/aholovko/compact-syntax-summaries-go/serializer/internal/summary"
)

var statFile = os.Stat

func main() {
	if err := runCommand(os.Args[1:], os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "go-syntax:", err)
		os.Exit(1)
	}
}

func runCommand(args []string, stdin io.Reader, stdout io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("missing subcommand: summarize")
	}

	if args[0] != "summarize" {
		return fmt.Errorf("unknown subcommand %q: expected summarize", args[0])
	}
	return runSummarize(args[1:], stdin, stdout)
}

func runSummarize(args []string, stdin io.Reader, stdout io.Writer) (err error) {
	fs := flag.NewFlagSet("summarize", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	inPath := fs.String("in", "", "input JSONL path (default: stdin)")
	outPath := fs.String("out", "", "output JSONL path (default: stdout)")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("summarize: unexpected argument %q", fs.Arg(0))
	}

	if *inPath != "" && *outPath != "" {
		inputInfo, statErr := statFile(*inPath)
		if statErr != nil {
			return statErr
		}
		outputInfo, statErr := statFile(*outPath)
		switch {
		case statErr == nil:
			if os.SameFile(inputInfo, outputInfo) {
				return fmt.Errorf("summarize: input and output refer to the same file")
			}
		case errors.Is(statErr, os.ErrNotExist):
			// A new output file is safe to create after the input has been opened.
		default:
			return statErr
		}
	}

	in := stdin
	if *inPath != "" {
		f, openErr := os.Open(*inPath)
		if openErr != nil {
			return openErr
		}
		defer func() {
			if closeErr := f.Close(); closeErr != nil && err == nil {
				err = closeErr
			}
		}()
		in = f
	}

	out := stdout
	if *outPath != "" {
		f, createErr := os.Create(*outPath)
		if createErr != nil {
			return createErr
		}
		defer func() {
			if closeErr := f.Close(); closeErr != nil && err == nil {
				err = closeErr
			}
		}()
		out = f
	}
	return summarizeJSONL(in, out)
}

func summarizeJSONL(in io.Reader, out io.Writer) (err error) {
	w := bufio.NewWriter(out)
	defer func() {
		if flushErr := w.Flush(); flushErr != nil && err == nil {
			err = flushErr
		}
	}()

	enc := json.NewEncoder(w)
	return scanJSONL(in, func(line []byte) error {
		var rec record.Input
		if err = json.Unmarshal(line, &rec); err != nil {
			return fmt.Errorf("decode input: %w", err)
		}
		if err = enc.Encode(summary.Summarize(rec.ID, rec.Code)); err != nil {
			return fmt.Errorf("encode output: %w", err)
		}
		return nil
	})
}

func scanJSONL(r io.Reader, fn func([]byte) error) error {
	br := bufio.NewReader(r)
	for {
		line, readErr := br.ReadBytes('\n')
		line = bytes.TrimRight(line, "\r\n")
		if len(bytes.TrimSpace(line)) > 0 {
			if err := fn(line); err != nil {
				return err
			}
		}

		if readErr != nil {
			if readErr != io.EOF {
				return readErr
			}
			return nil
		}
	}
}
