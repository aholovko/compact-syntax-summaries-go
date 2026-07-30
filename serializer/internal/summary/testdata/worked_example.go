package main

func ProcessItems(items []Item) error {
	for i := 0; i < len(items); i++ {
		if items[i].Status == "" {
			items[i].Status = "pending"
		}
	}
	return nil
}
