---
layout: home
title: Zakaria Alsahfi – Freelance Data Analyst & Data Visualization Portfolio
description: Freelance data analyst Zakaria Alsahfi showcases a powerful data visualization portfolio with Python, Excel, and real-world dashboards solving analytical challenges.
sitemap: true
cover: true
---

This website aims to compile and showcase captivating data visualizations and analyses. 
By sharing these insights with the world, I hope to provide valuable solutions and knowledge to benefit others.
{:.lead}


## Latest Posts

<!--posts-->

<!-- Welcome / About -->
<section id="about">
  {% include_relative welcome.html %}
</section>

<!-- Power BI -->
<section id="powerbi">
  <h2>Power BI Projects</h2>
  {% assign powerbi_pages = site.pages | where:"layout","powerbi" %}
  <div class="columns mt2 columns-break">
    {% for page in powerbi_pages %}
      <div class="column column-1-2">
        {% include_cached pro/project-card.html project=page %}
      </div>
    {% endfor %}
  </div>
</section>

<!-- Tableau -->
<section id="tableau">
  <h2>Tableau Dashboards</h2>
  {% assign tableau_pages = site.pages | where:"layout","tableau" %}
  <div class="columns mt2 columns-break">
    {% for page in tableau_pages %}
      <div class="column column-1-2">
        {% include_cached pro/project-card.html project=page %}
      </div>
    {% endfor %}
  </div>
</section>

<!-- Resume -->
<section id="resume">
  {% include_relative resume.html %}
</section>


See **[Posts](/posts/)** for more
