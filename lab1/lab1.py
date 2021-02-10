import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo.order_id.count()
    
    def info(self) -> None:
        # TODO
        # print data info.
        self.chipo.info()
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print(self.chipo.columns)
    
    def most_ordered_item(self):
        # TODO
        quantity = 0
        item_name = self.chipo['item_name'].value_counts().idxmax()
        order_id = self.chipo['order_id'].value_counts().idxmax()
        for i in range(len(self.chipo.order_id)):
            if self.chipo['item_name'].iloc[i] == item_name:
                quantity = quantity + self.chipo['quantity'].iloc[i]
        # quantity = self.chipo['quantity'].value_counts().idxmax()
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       return self.chipo['quantity'].sum()
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        # self.chipo['item_price'].apply(lambda row: float(row)).sum()
        self.chipo['item_price'] = self.chipo.item_price.apply(lambda x: float(x[1:]))
        return (self.chipo.item_price * self.chipo.quantity).sum()
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        return self.chipo['order_id'].iloc[-1]
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        avgSales = (self.chipo.item_price * self.chipo.quantity).sum() / self.chipo['order_id'].iloc[-1]
        return avgSales

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return self.chipo['item_name'].nunique()
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        graphDF = pd.DataFrame.from_dict(letter_counter, orient='index').reset_index()
        graphDF = graphDF.rename(columns = {'index' : 'item_name', 0 :'quantity'})
        graphDF = graphDF.sort_values(by = 'quantity', ascending = False)
        plt.figure(figsize=(12, 7))
        plt.bar(graphDF['item_name'][:5], height = graphDF.quantity[:5])
        plt.xlabel('Items', fontsize=18)
        plt.ylabel('Number of Orders', fontsize=16)
        plt.title('Most popular items', fontsize=20)
        plt.show(block=True)
        
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        groupedDF = self.chipo.groupby('order_id').agg({'item_price': 'sum', 'quantity':'sum'})
        print(groupedDF.head(10))
        plt.scatter(x=groupedDF.item_price, y=groupedDF.quantity, s = 50, c='blue')
        plt.show(block=True)
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()   #///////////////////
    assert item_name == 'Chicken Bowl'
    #assert order_id == 713926	
    #assert quantity == 159
    total = solution.total_item_orders() 
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    