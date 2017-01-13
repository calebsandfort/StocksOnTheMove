"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import numpy as np
import pandas as pd
import talib
from statsmodels import regression
import statsmodels.api as sm
import math
from scipy import stats

from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, CustomFactor
from quantopian.pipeline.filters.morningstar import Q500US
from quantopian.pipeline import factors, filters, classifiers
from quantopian.pipeline.data import morningstar
 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    
    context.spy = sid(8554)
    context.BUY_ALLOWED = False
    context.WEEK_COUNTER = 2
    context.RISK = .0015
    context.OUT_OF_CASH_THRESHOLD = 5
    context.printed = False
    
    schedule_function(do_stuff, date_rules.week_start(), time_rules.market_open())
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
     
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(), 'my_pipeline')

def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    
    # Base universe set to the Q500US
    #base_universe = Q600US()
    
    market_cap = MarketCap()
    top_N_market_cap = market_cap.top(500)

    is_primary_share = morningstar.share_class_reference.is_primary_share.latest
    is_not_adr = ~morningstar.share_class_reference.is_depositary_receipt.latest
    
    base_universe = (top_N_market_cap & is_primary_share & is_not_adr)
    
    stocks_on_the_move = StocksOnTheMoveFactor(mask=base_universe)
    momentum_rank = stocks_on_the_move.momentum.rank(ascending=False, mask=base_universe)

    pipe = Pipeline(
        screen = (top_N_market_cap & is_primary_share & is_not_adr),
        columns = {
            'momentum': stocks_on_the_move.momentum,
            'momentum_rank': momentum_rank,
            'above_100_sma': stocks_on_the_move.above_100_sma,
            'no_gaps': stocks_on_the_move.no_gaps
        }
    )
    return pipe

def Q600US():
    return filters.make_us_equity_universe(
        target_size=600,
        rankby=factors.AverageDollarVolume(window_length=200),
        mask=filters.default_us_equity_universe_mask(),
        groupby=classifiers.morningstar.Sector(),
        max_group_weight=1.0,
        smoothing_func=lambda f: f.downsample('month_start'),
    )

class MarketCap(CustomFactor):   
    
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]

class StocksOnTheMoveFactor(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 100
    
    outputs = ['momentum', 'above_100_sma', 'no_gaps']
    
    def compute(self, today, assets, out, data):
        momentum = []
        above_100_sma = []
        no_gaps = []
        no_gaps_bool = True
        last_90_days = []
        X = []
        Y = []
        model = None
        
        for col in data.T:
            try:
                last_90_days = col[-90:]
                
                if not np.isnan(last_90_days).any():
                    X = range(0, 90)
                    Y = np.log(last_90_days)
                    a_s,b_s,r,tt,stderr=stats.linregress(X,Y)

                    momentum.append(math.pow(1.0 + a_s, 250) * r * r)
                    
                    above_100_sma.append(col[-1] > np.mean(col))
                
                    no_gaps_bool = True

                    for i in range(1, 90):
                        if abs((last_90_days[i] - last_90_days[i-1])/last_90_days[i-1]) > .15:
                            no_gaps_bool = False
                            break

                    no_gaps.append(no_gaps_bool)
                else:
                    momentum.append(-1000.0)
                    above_100_sma.append(0.0)
                    no_gaps.append(0.0)
                
            except:
                momentum.append(-1000.0)
                above_100_sma.append(0.0)
                no_gaps.append(0.0)
                
        out.momentum[:] = momentum
        out.above_100_sma[:] = above_100_sma
        out.no_gaps[:] = no_gaps

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('my_pipeline')
  
    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index
    
    context.momentum_stocks = context.output["momentum"].copy()
    context.momentum_stocks.sort_values(ascending=False, inplace=True)
    
    context.above_100_sma = context.output.index[context.output['above_100_sma'] > 0.0]
    context.no_gaps = context.output.index[context.output['no_gaps'] > 0.0]
    context.top_100_by_momentum = context.output.index[context.output['momentum_rank'] <= 100]
    
    context.cash = context.portfolio.cash
    context.positions_and_cash = context.portfolio.cash + np.sum((context.portfolio.positions[position].amount * context.portfolio.positions[position].last_sale_price) for position in context.portfolio.positions)

def my_assign_weights(context, data):
    """
    Assign weights to securities that we want to order.
    """
    pass

def do_stuff(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    spy_hist = data.history(context.spy, 'price', 201, '1d')
    spy_sma = np.mean(spy_hist[:-1])
    context.BUY_ALLOWED = spy_hist[-1] > spy_sma
    out_of_cash_count = 0;
    
    #Liquidate unqualified
    for pos in context.portfolio.positions:
        if pos not in context.above_100_sma or pos not in context.no_gaps or pos not in context.top_100_by_momentum or pos not in context.momentum_stocks:
            current_position = context.portfolio.positions[pos]
            context.cash += current_position.last_sale_price * current_position.amount
            order_target_percent(pos, 0)
    
    cnt = 0
    
    #Buy until out of cash
    if context.BUY_ALLOWED:
        for stock in context.momentum_stocks.index:
            if stock not in context.portfolio.positions and stock in context.above_100_sma and stock in context.no_gaps and stock in context.top_100_by_momentum:
                shares = get_share_count(context, data, stock)
                cur_price = data.current(stock, 'price')
                purchase_amount = (shares * cur_price)
                
                if not context.printed and cnt < 10:
                    print "%s: %.6f" % (stock.symbol, context.momentum_stocks[stock])
                    cnt += 1

                if(purchase_amount < context.cash):
                    order(stock, shares)
                    context.cash -= purchase_amount
                else:
                    out_of_cash_count += 1
                
                if context.OUT_OF_CASH_THRESHOLD <= out_of_cash_count:
                    break
    context.printed = True
    
def get_share_count(context, data, stock):
    hist = data.history(stock, ['high', 'low', 'close'], 25, '1d')
    atr = talib.ATR(hist['high'],  hist['low'], hist['close'], timeperiod=20)
    
    return int((context.positions_and_cash * context.RISK)/atr[-2])
                
def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(leverage=context.account.leverage)
