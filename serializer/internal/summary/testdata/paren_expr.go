package main

func doWork() int { return 0 }

func run(ch chan int) {
	(doWork())
	(<-ch)
}
