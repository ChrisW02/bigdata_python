
import datetime as dt
import time as tm

# @formatter:on
# @formatter:off
print(tm.time())
dtnow = dt.datetime.fromtimestamp(tm.time())
print(dtnow)
print(dtnow.year,dtnow.month,dtnow.day,dtnow.hour,dtnow.minute,dtnow.second)

delta = dt.timedelta(days = 200)
today = dt.date.today()
print(today-delta)
print(today>today-delta)