---
layout: default
title: Zakaria Alsahfi – Freelance Data Analyst & Data Visualization Portfolio
description: Freelance data analyst Zakaria Alsahfi showcases a powerful data visualization portfolio with Python, Excel, and real-world dashboards solving analytical challenges.
sitemap: true
cover: true
---

This website aims to compile and showcase captivating data visualizations and analyses. 
By sharing these insights with the world, I hope to provide valuable solutions and knowledge to benefit others.
{:.lead}

---

## Power BI Projects
<ul>
{% for item in site.powerbi %}
  <li>
    <a href="{{ item.url | relative_url }}">{{ item.title }}</a>
  </li>
{% endfor %}
</ul>

---

## Tableau Dashboards
<ul>
{% for item in site.tableau %}
  <li>
    <a href="{{ item.url | relative_url }}">{{ item.title }}</a>
  </li>
{% endfor %}
</ul>

---

See **[Posts](/posts/)** for more
