package main

type Handler interface {
	Handle(r Request) error
	Close() error
}
