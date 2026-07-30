package main

func Wait(ch <-chan int) int {
	select {
	case v := <-ch:
		return v
	default:
		return 0
	}
}
