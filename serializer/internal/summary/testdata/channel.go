package main

func Interact(out chan<- int, in <-chan int) {
	out <- 1
	<-in
}
