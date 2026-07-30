package main

func Extract(i interface{}) {
	s, ok := i.(string)
	_, _ = s, ok
}
