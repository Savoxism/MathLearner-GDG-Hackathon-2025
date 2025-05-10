def calculate_large_posters_profit(sold, selling_price, cost_price):
    # Calculate the profit per large poster
    profit_per_large = selling_price - cost_price
    # Calculate total profit from large posters
    total_profit_large = sold * profit_per_large
    return total_profit_large

def calculate_small_posters_profit(sold, selling_price, cost_price):
    # Calculate the profit per small poster
    profit_per_small = selling_price - cost_price
    # Calculate total profit from small posters
    total_profit_small = sold * profit_per_small
    return total_profit_small

def calculate_total_profit(large_posters_per_day, small_posters_per_day, days):
    # Selling prices and costs
    large_selling_price = 10
    large_cost_price = 5
    small_selling_price = 6
    small_cost_price = 3
    
    # Calculate profits from large and small posters
    profit_large = calculate_large_posters_profit(large_posters_per_day * days, large_selling_price, large_cost_price)
    profit_small = calculate_small_posters_profit(small_posters_per_day * days, small_selling_price, small_cost_price)
    
    # Total profit
    total_profit = profit_large + profit_small
    return total_profit

def main():
    # Given data
    large_posters_per_day = 2
    small_posters_per_day = 3  # 5 - 2
    days = 5
    
    total_profit = calculate_total_profit(large_posters_per_day, small_posters_per_day, days)
    print(total_profit)

main()