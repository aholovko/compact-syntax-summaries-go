package main

func Describe(i interface{}) string {
	switch v := i.(type) {
	case int:
		return "int"
	default:
		_ = v
		return "other"
	}
}
