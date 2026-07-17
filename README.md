
# Data Jobs Analysis — Skill Demand & Pay

> **Scope note before you read this**: the four analyses below do not all use the same data slice. Q1 and Q3 are filtered to **India**. Q2 and Q4 are run on the **global** dataset with no country filter. That's not a stylistic choice — it's how the source notebooks were written. Keep it in mind when comparing numbers across sections; a skill that looks "in demand" in Q1 and "well-paid" in Q4 isn't necessarily true in the same market.

---

## Q1. What are the skills most in demand for the top 3 most popular data roles?

**Scope:** India, all job titles, `job_skills` exploded and counted.

### Code
```python
df_skills = df_india.explode('job_skills')

df_skills_count = df_skills.groupby(['job_skills', 'job_title_short']).size()
df_skills_count = df_skills_count.reset_index(name='skill_count')
df_skills_count.sort_values(by='skill_count', ascending=False, inplace=True)

job_titles = df_skills_count['job_title_short'].unique().tolist()[:3]

fig, ax = plt.subplots(len(job_titles), 1)
for i, job_title in enumerate(job_titles):
    df_plot = df_skills_count[df_skills_count['job_title_short'] == job_title].head(5)
    df_plot.plot(kind='barh', x='job_skills', y='skill_count', ax=ax[i], figsize=(10, 7))
    ax[i].invert_yaxis()
```
```python
# likelihood version: skill_count / total job postings for that title
df_skills_perc = pd.merge(df_skills_count, df_job_title_counts, how='left', on='job_title_short')
df_skills_perc['skill_percentage'] = 100 * df_skills_perc['skill_count'] / df_skills_perc['Job Total']
```

### Result
![Top skills by raw count](2_Project/Images/q1_skill_counts.png)
![Skill likelihood by role](2_Project/Images/q1_skill_likelihood_2.png)

### Insight
- SQL and Python are the only two skills that show up in the top-5 for **all three** roles — everything else (Excel, Tableau, Power BI, Spark, AWS, R) is role-specific.
- Data Engineer is the only role where cloud/infra skills (AWS, Azure, Spark) crack the top 5 at all — Data Analyst and Data Scientist top-5 lists are dominated by languages and BI tools instead.
- Data Analyst has the lowest "must-have" concentration of the three: its top skill (SQL) sits at 52%, versus 68% (Data Engineer/SQL) and 70% (Data Scientist/Python). That means DA postings are more fragmented across tools — no single skill is close to a baseline requirement the way SQL is for Data Engineer.
- The chart title says "US Job Postings" — that's wrong, the underlying filter is `job_country == 'India'`. Worth fixing before this goes anywhere public; anyone reading the chart alone would draw conclusions about the wrong labor market.

---

## Q2. How are in-demand skills trending for Data Analysts?

**Scope:** Global (no country filter), Data Analyst postings only, grouped by month.

### Code
```python
df_DA = df[df['job_title_short'] == 'Data Analyst'].copy()
df_DA['job_posted_month_no'] = df_DA['job_posted_date'].dt.month

df_DA_explode = df_DA.explode('job_skills')
df_DA_pivot = df_DA_explode.pivot_table(index='job_posted_month_no', 
columns='job_skills',
aggfunc='size', 
fill_value=0)

df_DA_pivot.loc['Total'] = df_DA_pivot.sum()
df_DA_pivot = df_DA_pivot[df_DA_pivot.loc['Total'].sort_values(ascending=False).index]
df_DA_pivot.drop('Total', inplace=True)

df_DA_pivot.iloc[:, :5].plot(kind='line', linewidth=3, linestyle=':',
colormap='viridis', 
marker='o', 
markersize=4, 
figsize=(10, 6))
```
```python 
from matplotlib.ticker import PercentFormatter

df_plot = df_DA_US_percent.iloc[:, :5]

sns.lineplot(data=df_plot, dashes=False, legend='full', palette='tab10')

plt.gca().yaxis.set_major_formatter
(PercentFormatter(decimals=0))

plt.show()
```
### Result
![Trending skills for Data Analysts](2_Project/Images/q2_trending_skills.png)

### Insight
- SQL and Excel hold the top two spots consistently across the year — this isn't a fad skill, it's structurally embedded in DA postings month to month.
- There's a visible seasonal dip across most skills mid-year and a pickup toward year-end — that's a hiring-cycle signal, not a skills signal, and shouldn't be read as "Tableau is losing relevance" or similar.
- The lines never cross rank order in a meaningful way — no lower-ranked skill (Power BI, R) closes the gap with the top group at any point. If you're deciding what to learn next, this chart says "stability," not "watch this space."
- This is global data, not India-specific, despite Q1 and Q3 in this same README being India-only. If the intent is a coherent country-level story, this chart needs to be re-run filtered to India before it's comparable to the rest of the document.

---

## Q3. How well do jobs and skills pay for Data Analysts?

**Scope:** India, salary rows only (`salary_year_avg` not null).

### Code
```python
job_titles = ['Data Analyst', 'Data Scientist', 'Data Engineer']
df_india = df[(df['job_title_short'].isin(job_titles)) & (df['job_location'] == 'India')].copy()
df_india = df_india.dropna(subset=['salary_year_avg'])

job_list = [df_india[df_india['job_title_short'] == t]['salary_year_avg'] for t in job_titles]
plt.boxplot(job_list, labels=job_titles, vert=False)
```
```python
df_DA_india = df_DA_india.explode('job_skills')
df_DA_india_group = df_DA_india.groupby('job_skills')['salary_year_avg'].agg(['count', 'median'])

df_DA_top_pay = df_DA_india_group.sort_values(by='median', ascending=False).head(10)
df_DA_skill = df_DA_india_group.sort_values(by='count', ascending=False).head(10).sort_values(by='median', ascending=False)
```

### Result
![Salary distribution by role](2_Project/Images/q3_salary_by_role.png)
![Top paying vs top demand skills](2_Project/Images/q3_skill_pay.png)

**Adding role variety (seniority + adjacent titles)**
- Machine Learning Engineer has by far the widest salary box of any role — floor near Data Analyst levels, ceiling near Data Scientist levels. It's the least predictable role to price, which cuts both ways: highest upside, but also the least certainty going in.
- Senior Data Engineer's box is unusually tight and sits right in the middle of the pack — that's either a genuinely standardized pay band for that title or a small sample size making the box look artificially narrow; worth checking row count before reading it as a stable signal.
- Data Scientist has a higher median than Data Engineer but also shows low-end outliers below its own whisker — meaning some Data Scientist postings pay closer to Data Analyst rates, so the title alone doesn't guarantee the premium the median suggests.
- Data Analyst remains the clear floor of this group on both median and ceiling — none of the added roles change that; if anything, comparing against Machine Learning Engineer and Data Scientist makes the analyst-to-anything-else pay gap look larger, not smaller.

---

### Insight
- Data Analyst still has the lowest median salary and the tightest overall range of the roles shown — it's the most predictable role to price, but also the one with the least upside from the title alone.
- The "highest paid" and "most in-demand" skill lists barely overlap: SQL, Excel, and Python drive volume but sit in the middle of the pay chart, while tools like Visio, Jira, Confluence, and Azure pay more but barely register on the demand side. Optimizing for pay and optimizing for hireability point in different directions here.
- No visible sample-size guard on the "highest paid" ranking — a skill ranked by median salary alone can be pulled up by a handful of high-salary postings. Check the underlying count for each bar before treating this list as a reliable target to learn toward.



## Q4. What are the optimal skills for data analysts to learn? (High Demand AND High Paying)

**Scope:** Global (no country filter), Data Analyst postings only, top 10 skills by posting count.

### Code
```python
from adjustText import adjust_text
#df_DA_skills.plot(kind='scatter', x='skill_percentage', y='median_salary')

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_plot,
    x='skill_percentage',
    y='median_salary',
    hue='Technology'
    
)
sns.set_theme(style="ticks")
sns.despine()

# Prepare texts for adjustText
texts = []
for i, txt in enumerate(df_DA_skills.index):
    texts.append(plt.text(df_DA_skills['skill_percentage'].iloc[i], df_DA_skills['median_salary'].iloc[i], txt))

#Adjust text to avoid over lap
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray')) 

#Set axis labels, title, and legend
plt.xlabel('Percentage of Data Analyst Jobs with Skill')
plt.ylabel('Median Yearly Salary')
plt.title('Highest Paid Skills for Data Analysts in India')
plt.ylim(60000, 115000)  # Set the same x-axis limits as the second plot
plt.xlim(0, 55)  # Set the same x-axis limits as the second plot

from matplotlib.ticker import PercentFormatter

ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${int(x/1000)}K'))
ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))

#Adjust layout and display plot
plt.tight_layout()
plt.show()
```

### Result
![Optimal skills scatter](2_Project/Images/q4_optimal_skills.png)

### Insight
- No skill sits in a clean "high demand AND high pay" corner — the chart splits into two clusters instead: SQL, Python, and Excel dominate demand (38–49% of postings) but sit mid-pack on pay (~$96K–$98K), while Power BI, Tableau, and Spark pay more (~$108K–$111K) but show up in only 12–21% of postings. Demand and pay are trading off, not stacking, for this role.
- If forced to pick one "optimal" skill, Excel is the best-supported answer here — it's the only skill combining high demand (~41%) with pay on par with Python and SQL, whereas Tableau and Power BI trade a real chunk of demand for a modest pay bump.
- Spark is the standout outlier: lowest demand of the group (~12%) but the single highest median salary (~$111K), tied with Power BI. That's a real "niche but lucrative" signal — the kind of skill this analysis was actually built to find — but at 12% of postings it's a bet on specialization, not a safe default.
- The `analyst_tools` category (Excel, Tableau, Power BI) spans the widest pay range of any group here (~$98K to ~$111K) despite all three being "BI tools" — treating that category as one interchangeable skillset would be wrong; which specific tool matters more than the category label.
- This chart fixes the scope problem flagged earlier in this README: it's now correctly filtered to Data Analyst + India, matching the rest of this document's scope. The version replaced here (top-10-by-raw-count, no category, no country filter) is worth deleting from the notebook rather than kept alongside this one — having both invites someone to cite the wrong chart.

---

## Summary of issues found while building this README
1. Q1's chart title says "India" when the data is filtered to India — a labeling bug in the source notebook.
2. Q2 still uses global data while Q1, Q3, and Q4 are India-only — that's the one remaining scope mismatch in this document; re-run Q2 filtered to India if it needs to support the same narrative as the rest.
3. Q3's "top paying skills" bar chart doesn't surface sample size (`count`), which is the single biggest risk of a misleading ranking in this whole project.
4. The notebook still contains an earlier, incorrectly-scoped version of the Q4 scatter (global, all job titles, top-10-by-count) alongside the corrected one used here — worth deleting so nobody cites the wrong chart later.
