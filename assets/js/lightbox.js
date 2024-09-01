function is_youtubelink(url) {
  var p =
    /^(?:https?:\/\/)?(?:www\.)?(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))((\w|-){11})(?:\S+)?$/;
  return url.match(p) ? RegExp.$1 : false;
}
function is_imagelink(url) {
  var p = /([a-z\-_0-9\/\:\.]*\.(jpg|jpeg|png|gif|webp))/i;
  return url.match(p) ? true : false;
}
function setGallery(el) {
  var elements = document.body.querySelectorAll(".gallery");
  elements.forEach((element) => {
    element.classList.remove("gallery");
  });
  if (el.closest("ul, p")) {
    var link_elements = el
      .closest("ul, p")
      .querySelectorAll("a[class*='lightbox-']");
    link_elements.forEach((link_element) => {
      link_element.classList.remove("current");
    });
    link_elements.forEach((link_element) => {
      if (el.getAttribute("href") == link_element.getAttribute("href")) {
        link_element.classList.add("current");
      }
    });
    if (link_elements.length > 1) {
      document.getElementById("lightbox").classList.add("gallery");
      link_elements.forEach((link_element) => {
        link_element.classList.add("gallery");
      });
    }
    var currentkey;
    var gallery_elements = document.querySelectorAll("a.gallery");
    Object.keys(gallery_elements).forEach(function (k) {
      if (gallery_elements[k].classList.contains("current")) currentkey = k;
    });
    if (currentkey == gallery_elements.length - 1) var nextkey = 0;
    else var nextkey = parseInt(currentkey) + 1;
    if (currentkey == 0) var prevkey = parseInt(gallery_elements.length - 1);
    else var prevkey = parseInt(currentkey) - 1;
    document.getElementById("next").addEventListener("click", function () {
      gallery_elements[nextkey].click();
    });
    document.getElementById("prev").addEventListener("click", function () {
      gallery_elements[prevkey].click();
    });
  }
}

function apply_lightbox_to_img_tag() {
  Array.from(document.querySelectorAll("img")).forEach((img_element) => {
    if (img_element.classList.contains("no-lightbox")) {
      return;
    }
    var wrapper = document.createElement("a");
    var url = img_element.getAttribute("src");
    wrapper.href = url;
    if (is_youtubelink(url)) {
      wrapper.classList.add("lightbox-youtube");
      wrapper.setAttribute("data-id", is_youtubelink(url));
    } else if (is_imagelink(url)) {
      wrapper.classList.add("lightbox-image");
      var href = wrapper.getAttribute("href");
      var filename = href.split("/").pop();
      var name = filename.split(".")[0];
      wrapper.setAttribute("title", name);
    } else {
      return;
    }
    img_element.parentNode.insertBefore(wrapper, img_element);
    wrapper.appendChild(img_element);
  });
}

function apply_lightbox() {
  var newdiv = document.createElement("div");
  newdiv.setAttribute("id", "lightbox");
  document.body.appendChild(newdiv);

  apply_lightbox_to_img_tag();

  // remove the clicked lightbox
  document
    .getElementById("lightbox")
    .addEventListener("click", function (event) {
      if (event.target.id != "next" && event.target.id != "prev") {
        this.innerHTML = "";
        document.getElementById("lightbox").style.display = "none";
      }
    });

  //add the youtube lightbox on click
  document.querySelectorAll("a.lightbox-youtube").forEach((element) => {
    element.addEventListener("click", function (event) {
      event.preventDefault();
      document.getElementById("lightbox").innerHTML =
        '<a id="close"></a><a id="next">&rsaquo;</a><a id="prev">&lsaquo;</a><div class="videoWrapperContainer"><div class="videoWrapper"><iframe src="https://www.youtube.com/embed/' +
        this.getAttribute("data-id") +
        '?autoplay=1&showinfo=0&rel=0"></iframe></div>';
      document.getElementById("lightbox").style.display = "block";

      setGallery(this);
    });
  });
}

/**
 * @license
 * Copyright 2019 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {Audit} from '../audit.js';
import * as i18n from '../../lib/i18n/i18n.js';
import {LargestContentfulPaint as ComputedLcp} from '../../computed/metrics/largest-contentful-paint.js';

const UIStrings = {
  /** Description of the Largest Contentful Paint (LCP) metric, which marks the time at which the largest text or image is painted by the browser. This is displayed within a tooltip when the user hovers on the metric name to see more. No character length limits. The last sentence starting with 'Learn' becomes link text to additional documentation. */
  description: 'Largest Contentful Paint marks the time at which the largest text or image is ' +
      `painted. [Learn more about the Largest Contentful Paint metric](https://developer.chrome.com/docs/lighthouse/performance/lighthouse-largest-contentful-paint/)`,
};

const str_ = i18n.createIcuMessageFn(import.meta.url, UIStrings);

class LargestContentfulPaint extends Audit {
  /**
   * @return {LH.Audit.Meta}
   */
  static get meta() {
    return {
      id: 'largest-contentful-paint',
      title: str_(i18n.UIStrings.largestContentfulPaintMetric),
      description: str_(UIStrings.description),
      scoreDisplayMode: Audit.SCORING_MODES.NUMERIC,
      supportedModes: ['navigation'],
      requiredArtifacts: ['HostUserAgent', 'traces', 'devtoolsLogs', 'GatherContext', 'URL'],
    };
  }

  /**
   * @return {{mobile: {scoring: LH.Audit.ScoreOptions}, desktop: {scoring: LH.Audit.ScoreOptions}}}
   */
  static get defaultOptions() {
    return {
      mobile: {
        // 25th and 13th percentiles HTTPArchive -> median and p10 points.
        // https://bigquery.cloud.google.com/table/httparchive:lighthouse.2020_02_01_mobile?pli=1
        // https://web.dev/articles/lcp#what_is_a_good_lcp_score
        // see https://www.desmos.com/calculator/1etesp32kt
        scoring: {
          p10: 2500,
          median: 4000,
        },
      },
      desktop: {
        // 25th and 5th percentiles HTTPArchive -> median and p10 points.
        // SELECT
        //   APPROX_QUANTILES(lcpValue, 100)[OFFSET(5)] AS p05_lcp,
        //   APPROX_QUANTILES(lcpValue, 100)[OFFSET(25)] AS p25_lcp
        // FROM (
        //   SELECT CAST(JSON_EXTRACT_SCALAR(payload, "$['_chromeUserTiming.LargestContentfulPaint']") AS NUMERIC) AS lcpValue
        //   FROM `httparchive.pages.2020_04_01_desktop`
        // )
        scoring: {
          p10: 1200,
          median: 2400,
        },
      },
    };
  }

  /**
   * @param {LH.Artifacts} artifacts
   * @param {LH.Audit.Context} context
   * @return {Promise<LH.Audit.Product>}
   */
  static async audit(artifacts, context) {
    const trace = artifacts.traces[Audit.DEFAULT_PASS];
    const devtoolsLog = artifacts.devtoolsLogs[Audit.DEFAULT_PASS];
    const gatherContext = artifacts.GatherContext;
    const metricComputationData = {trace, devtoolsLog, gatherContext,
      settings: context.settings, URL: artifacts.URL};

    const metricResult = await ComputedLcp.request(metricComputationData, context);
    const options = context.options[context.settings.formFactor];

    return {
      score: Audit.computeLogNormalScore(
        options.scoring,
        metricResult.timing
      ),
      scoringOptions: options.scoring,
      numericValue: metricResult.timing,
      numericUnit: 'millisecond',
      displayValue: str_(i18n.UIStrings.seconds, {timeInMs: metricResult.timing}),
    };
  }
}

export default LargestContentfulPaint;
export {UIStrings};
