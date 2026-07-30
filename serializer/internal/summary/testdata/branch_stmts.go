package main

func classify(xs []int) int {
	for i := range xs {
		if xs[i] < 0 {
			continue
		}
		if xs[i] == 0 {
			break
		}
	}
	switch xs[0] {
	case 1:
		fallthrough
	case 2:
	}
	return 0
}
