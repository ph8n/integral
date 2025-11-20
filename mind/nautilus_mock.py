from dataclasses import dataclass, field
from typing import List, Dict, Optional
from decimal import Decimal


class OrderSide:
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce:
    GTC = "GTC"


@dataclass
class InstrumentId:
    value: str

    @staticmethod
    def from_str(s: str):
        return InstrumentId(value=s)

    def __str__(self):
        return self.value


@dataclass
class Instrument:
    id: InstrumentId

    def make_qty(self, qty):
        return qty


@dataclass
class Bar:
    instrument_id: InstrumentId
    close: float
    ts_event: int  # timestamp


@dataclass
class StrategyConfig:
    instrument_ids: List[str]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Logger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")


class Cache:
    def __init__(self):
        self.bars = {}

    def bar(self, instrument_id):
        return self.bars.get(str(instrument_id))


class Position:
    def __init__(self, quantity=0):
        self.quantity = quantity


class Account:
    def __init__(self):
        self.equity_total = Decimal("100000.0")


class Portfolio:
    def __init__(self):
        self.positions = {}
        self.account_obj = Account()

    def position(self, instrument_id):
        return self.positions.get(str(instrument_id), Position(0))

    def account(self, venue_id):
        return self.account_obj


class OrderFactory:
    def market(self, instrument_id, order_side, quantity, time_in_force):
        return {
            "instrument_id": instrument_id,
            "side": order_side,
            "quantity": quantity,
            "time_in_force": time_in_force,
        }


class Strategy:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.log = Logger()
        self.cache = Cache()
        self.portfolio = Portfolio()
        self.order_factory = OrderFactory()
        self.venue_id = "SIM"
        self.orders = []

    def on_start(self):
        pass

    def on_bar(self, bar: Bar):
        pass

    def on_stop(self):
        pass

    def subscribe_bars(self, instrument_id):
        print(f"Subscribed to {instrument_id}")

    def unsubscribe_bars(self, instrument_id):
        pass

    def instrument(self, instrument_id):
        return Instrument(id=instrument_id)

    def submit_order(self, order):
        self.orders.append(order)
        self.log.info(
            f"Order Submitted: {order['side']} {order['quantity']} {order['instrument_id']}"
        )
