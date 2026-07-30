package main

func Run(ch chan int) {
	defer close(ch)
	go work()
}
