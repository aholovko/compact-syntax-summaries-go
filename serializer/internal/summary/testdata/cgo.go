package main

import "C"

func Hello() {
	C.puts(C.CString("hello"))
}
