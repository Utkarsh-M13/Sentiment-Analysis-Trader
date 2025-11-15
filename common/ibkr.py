from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

contract = Stock('SPY', 'SMART', 'USD')

order = MarketOrder('BUY', 5)
trade = ib.placeOrder(contract, order)

print("Connected:", ib.isConnected())
