"""Deliberately weak map-free baseline for a restricted lap task."""

packet = get_scan()
while not packet["done"]:
    packet = drive(0.0, 1.0, steps=2)

RESULT = {"finished": packet["done"]}
