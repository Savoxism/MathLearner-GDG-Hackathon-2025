def donation_first_km(x):
    return x

def donation_second_km(x):
    return donation_first_km(x) * 2

def donation_third_km(x):
    return donation_second_km(x) * 2

def donation_fourth_km(x):
    return donation_third_km(x) * 2

def donation_fifth_km(x):
    return donation_fourth_km(x) * 2

def total_donation(x):
    return (donation_first_km(x) +
            donation_second_km(x) +
            donation_third_km(x) +
            donation_fourth_km(x) +
            donation_fifth_km(x))

def find_x(total_amount):
    for x in range(1, total_amount + 1):
        if total_donation(x) == total_amount:
            return x
    return None

total_amount = 310
x_value = find_x(total_amount)
print(x_value)