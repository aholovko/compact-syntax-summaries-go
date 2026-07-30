package p

func f(x int) {
	y := /* set */ 1
	if /* why */ x > 0 {
		g(y) // trailing
	}
}
