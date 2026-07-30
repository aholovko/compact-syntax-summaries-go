// Package allowlist defines safe serializer-authored vocabulary.
package allowlist

import "strings"

// Roles is the complete vocabulary allowed in serializer-authored segments.
var Roles = []string{
	"func", "type", "const", "var",
	"struct", "interface", "=",
	"loop:", "for", "range", "index", "condition", "update", "key", "value", "over",
	"conditional:", "else", "switch:", "case:", "default:", "select:",
	"defer:", "go:", "return:",
	"assignment:", "access:", "call:", "composite-literal:", "channel:",
	"send", "receive",
	"comment:", "line", "block", "doc",
	"types/annotations:",
}

var roleSet = func() map[string]bool {
	m := make(map[string]bool, len(Roles))
	for _, r := range Roles {
		m[r] = true
	}
	return m
}()

// Tokenize splits authored text into role tokens for allowlist validation.
func Tokenize(authored string) []string {
	fields := strings.FieldsFunc(authored, func(r rune) bool {
		return r == ' ' || r == ';' || r == ',' || r == '(' || r == ')'
	})
	return fields
}

// Validate rejects authored tokens outside Roles.
func Validate(authoredTokens []string) (bad string, ok bool) {
	for _, tok := range authoredTokens {
		if !roleSet[tok] {
			return tok, false
		}
	}
	return "", true
}
