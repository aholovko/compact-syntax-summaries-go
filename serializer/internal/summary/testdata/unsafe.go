package main

import "unsafe"

func Size(x int) uintptr {
	return unsafe.Sizeof(x)
}
