import statsmodels.api as sm

from statsmodels.formula.api import ols

moore = sm.datasets.get_rdataset("Moore", "carData",
                                  cache=True) # load data
 

data = moore.data

data = data.rename(columns={"partner.status":
                              "partner_status"})


moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
                 data=data).fit()
 

table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 ANOVA DataFrame

print(
   table.sort_values("sum_sq", ascending=False).reset_index().rename(columns={"index": "Factor"})
)
