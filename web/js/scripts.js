console.log(document.title);

// =========================================
// FIREBASE CONFIGURATION
// =========================================
const firebaseConfig = {
  apiKey: "AIzaSyCgXIk7OAq5pAMVV1oYogznt_Gc2BCB8KM",
  authDomain: "banana-sorting-5f22d.firebaseapp.com",
  databaseURL: "https://banana-sorting-5f22d-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "banana-sorting-5f22d",
  storageBucket: "banana-sorting-5f22d.firebasestorage.app",
  messagingSenderId: "552716690697",
  appId: "1:552716690697:web:4590775f24e701d7cbdfd8",
  measurementId: "G-THNGNCEGMF"
};

if (!firebase.apps.length) {
  firebase.initializeApp(firebaseConfig);
}
const db = firebase.firestore();

// =========================================
// CHART INSTANCES
// =========================================
let pieChart, monthlyChart;

// =========================================
// CACHED RECORDS + ACTIVE FILTER STATE
// =========================================
let cachedRecords = [];

// Stores the last applied filter so the 3-second refresh re-applies
// it instead of reverting to the default Jan-Dec monthly view.
let activeFilter = null;

// Stores the latest period counts so the dropdown can switch without re-fetching
let periodData = { today: {count:0,subText:''}, week: {count:0,subText:''}, month: {count:0,subText:''}, year: {count:0,subText:''} };
let activePeriod = 'today';

function renderPeriodCount() {
  const pd = periodData[activePeriod];
  document.getElementById('periodCount').innerText   = pd.count;
  document.getElementById('periodSubText').innerText = pd.subText;
}

// =========================================
// DATE / TIMESTAMP HELPERS
// =========================================

function parseTimestamp(v) {
  if (v && typeof v.toDate === 'function') return v.toDate();
  if (typeof v === 'string')               return new Date(v.replace(' ', 'T'));
  return new Date(v);
}

function parseDateLocal(str) {
  const [y, m, d] = str.split('-').map(Number);
  const dt = new Date(y, m - 1, d);
  dt.setHours(0, 0, 0, 0);
  return dt;
}

function toLocalDateKey(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function startOfDay(date)  { const d = new Date(date); d.setHours(0,0,0,0);       return d; }
function endOfDay(date)    { const d = new Date(date); d.setHours(23,59,59,999);   return d; }

function startOfWeek(date) {
  const d = new Date(date), day = d.getDay();
  d.setDate(d.getDate() + (day === 0 ? -6 : 1 - day));
  d.setHours(0,0,0,0); return d;
}
function endOfWeek(date) {
  const e = new Date(startOfWeek(date));
  e.setDate(e.getDate() + 6); e.setHours(23,59,59,999); return e;
}

function startOfMonth(date) {
  const d = new Date(date.getFullYear(), date.getMonth(), 1);
  d.setHours(0,0,0,0); return d;
}
function endOfMonth(date) {
  const d = new Date(date.getFullYear(), date.getMonth()+1, 0);
  d.setHours(23,59,59,999); return d;
}
function startOfYear(date) {
  const d = new Date(date.getFullYear(), 0, 1);
  d.setHours(0,0,0,0); return d;
}
function endOfYear(date) {
  const d = new Date(date.getFullYear(), 11, 31);
  d.setHours(23,59,59,999); return d;
}

// =========================================
// Y-AXIS  0 → 100  in steps of 10
// =========================================
const yAxisInteger = {
  labels: { formatter: val => Math.floor(val).toString() },
  min: 0,
  max: 100,
  tickAmount: 10
};

// =========================================
// POPULATE YEAR DROPDOWNS
// =========================================
function populateYearDropdowns() {
  const currentYear = new Date().getFullYear();
  const startYearSel = document.getElementById('start-month-year');
  const endYearSel   = document.getElementById('end-month-year');

  for (let y = currentYear - 5; y <= currentYear + 5; y++) {
    const opt1 = document.createElement('option');
    opt1.value = y; opt1.textContent = y;
    if (y === currentYear) opt1.selected = true;
    startYearSel.appendChild(opt1);

    const opt2 = document.createElement('option');
    opt2.value = y; opt2.textContent = y;
    if (y === currentYear) opt2.selected = true;
    endYearSel.appendChild(opt2);
  }
}

// =========================================
// FIRESTORE FETCH + DASHBOARD REFRESH
// =========================================
async function updateDashboard() {
  try {
    const snapshot = await db.collection("banana_records").get();
    const allRecords = [];
    const now = new Date();

    snapshot.forEach(doc => {
      const data = doc.data();
      allRecords.push({ ...data, timestamp: parseTimestamp(data.timestamp) });
    });
    cachedRecords = allRecords;

    const tS = startOfDay(now),   tE = endOfDay(now);
    const wS = startOfWeek(now),  wE = endOfWeek(now);
    const mS = startOfMonth(now), mE = endOfMonth(now);
    const yS = startOfYear(now),  yE = endOfYear(now);

    const total      = allRecords.length;
    const todayCnt   = allRecords.filter(r => r.timestamp >= tS && r.timestamp <= tE).length;
    const weeklyCnt  = allRecords.filter(r => r.timestamp >= wS && r.timestamp <= wE).length;
    const monthlyCnt = allRecords.filter(r => r.timestamp >= mS && r.timestamp <= mE).length;
    const yearlyCnt  = allRecords.filter(r => r.timestamp >= yS && r.timestamp <= yE).length;

    periodData = {
      today: { count: todayCnt,   subText: now.toLocaleDateString() },
      week:  { count: weeklyCnt,  subText: `${wS.toLocaleDateString()} - ${wE.toLocaleDateString()}` },
      month: { count: monthlyCnt, subText: now.toLocaleDateString(undefined, { month: 'long', year: 'numeric' }) },
      year:  { count: yearlyCnt,  subText: now.getFullYear().toString() }
    };

    // GOOD / BAD cards show TODAY's data only
    const todayRecords = allRecords.filter(r => r.timestamp >= tS && r.timestamp <= tE);
    const good = todayRecords.filter(r => r.grade && r.grade.toUpperCase() === 'GOOD').length;
    const bad  = todayRecords.filter(r => r.grade && r.grade.toUpperCase() === 'BAD').length;

    document.getElementById('totalInspections').innerText = total;
    document.getElementById('goodCount').innerText        = good;
    document.getElementById('badCount').innerText         = bad;

    renderPeriodCount();
    updatePieChart(good, bad);

    if (activeFilter) {
      applyFilterToChart(activeFilter.type, activeFilter.startVal, activeFilter.endVal);
    } else {
      updateDefaultLineChart(allRecords);
    }

  } catch (err) {
    console.error("Dashboard error:", err);
  }
}

// =========================================
// CHART RENDER / UPDATE
// =========================================

function updatePieChart(good, bad) {
  const opts = {
    series: [good, bad],
    chart: {
      type: 'donut',
      height: 260,
      animations: { enabled: true, easing: 'easeinout', speed: 800 }
    },
    labels: ['GOOD', 'BAD'],
    colors: ['#28a745', '#dc3545'],
    legend: { position: 'bottom', fontSize: '14px', labels: { useSeriesColors: true } },
    dataLabels: {
      enabled: true,
      formatter: val => val.toFixed(1) + '%',
      style: { fontSize: '14px', fontWeight: 'bold', colors: ['#000'] },
      background: { enabled: true, foreColor: '#fff', padding: 6, borderRadius: 6, opacity: 0.8 }
    },
    fill: {
      type: 'gradient',
      gradient: {
        shade: 'light', type: 'horizontal', shadeIntensity: 0.5,
        gradientToColors: ['#82e0aa', '#f1948a'], inverseColors: true,
        opacityFrom: 0.9, opacityTo: 0.9, stops: [0, 100]
      }
    },
    dropShadow: { enabled: true, top: 2, left: 2, blur: 4, opacity: 0.2 },
    plotOptions: {
      pie: {
        donut: { size: '60%' }
      }
    }
  };
  if (pieChart) { pieChart.updateSeries([good, bad]); }
  else { pieChart = new ApexCharts(document.querySelector("#pie-chart"), opts); pieChart.render(); }
}

// Default: Jan-Dec overview for all records
function updateDefaultLineChart(allRecords) {
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const data   = new Array(12).fill(0);
  allRecords.forEach(r => { data[r.timestamp.getMonth()]++; });

  const opts = {
    series: [{ name: 'Bananas Sorted', data }],
    chart: {
      type: 'line', height: 260,
      toolbar: { show: true, tools: { download:true, zoom:true, zoomin:true, zoomout:true, pan:true, reset:true } }
    },
    xaxis: { categories: months },
    yaxis: yAxisInteger,
    colors: ['#ffc107'],
    stroke: { curve: 'smooth' }
  };

  if (monthlyChart) {
    monthlyChart.updateOptions({ xaxis: { categories: months }, yaxis: yAxisInteger });
    monthlyChart.updateSeries([{ name: 'Bananas Sorted', data }]);
  } else {
    monthlyChart = new ApexCharts(document.querySelector("#monthly-chart"), opts);
    monthlyChart.render();
  }
}

// =========================================
// FILTER LOGIC
// =========================================
function applyFilterToChart(type, startVal, endVal) {
  const categories = [];
  const data       = [];

  if (type === 'daily') {
    const start = parseDateLocal(startVal);
    const end   = parseDateLocal(endVal);
    end.setHours(23, 59, 59, 999);
    const dayMap = {};
    cachedRecords
      .filter(r => r.timestamp >= start && r.timestamp <= end)
      .forEach(r => {
        const key = toLocalDateKey(r.timestamp);
        dayMap[key] = (dayMap[key] || 0) + 1;
      });
    const cursor = new Date(start);
    while (cursor <= end) {
      const key = toLocalDateKey(cursor);
      categories.push(key);
      data.push(dayMap[key] || 0);
      cursor.setDate(cursor.getDate() + 1);
    }

  } else if (type === 'weekly') {
    const start = parseDateLocal(startVal);
    const end   = parseDateLocal(endVal);
    end.setHours(23, 59, 59, 999);
    const cursor = new Date(start);
    let weekNo = 0;
    while (cursor <= end) {
      const wS = new Date(cursor);
      const wE = new Date(cursor);
      wE.setDate(wE.getDate() + 6);
      wE.setHours(23, 59, 59, 999);
      const count = cachedRecords.filter(r => r.timestamp >= wS && r.timestamp <= wE).length;
      categories.push(`Wk${weekNo + 1} (${toLocalDateKey(wS)})`);
      data.push(count);
      cursor.setDate(cursor.getDate() + 7);
      weekNo++;
    }

  } else if (type === 'monthly') {
    let y = startVal.year, m = startVal.month;
    const endY = endVal.year, endM = endVal.month;
    while (y < endY || (y === endY && m <= endM)) {
      const mS = new Date(y, m - 1, 1); mS.setHours(0, 0, 0, 0);
      const mE = new Date(y, m, 0);     mE.setHours(23, 59, 59, 999);
      const count = cachedRecords.filter(r => r.timestamp >= mS && r.timestamp <= mE).length;
      categories.push(mS.toLocaleString(undefined, { month: 'short', year: 'numeric' }));
      data.push(count);
      m++; if (m > 12) { m = 1; y++; }
    }

  } else if (type === 'yearly') {
    const startY = parseInt(startVal);
    const endY   = parseInt(endVal);
    for (let y = startY; y <= endY; y++) {
      const yS = new Date(y, 0, 1);   yS.setHours(0, 0, 0, 0);
      const yE = new Date(y, 11, 31); yE.setHours(23, 59, 59, 999);
      const count = cachedRecords.filter(r => r.timestamp >= yS && r.timestamp <= yE).length;
      categories.push(y.toString());
      data.push(count);
    }
  }

  if (monthlyChart) {
    monthlyChart.updateOptions({ xaxis: { categories }, yaxis: yAxisInteger });
    monthlyChart.updateSeries([{ name: 'Bananas Sorted', data }]);
  }
}

// =========================================
// FILTER UI  (DOM wiring)
// =========================================
document.addEventListener('DOMContentLoaded', function () {

  populateYearDropdowns();

  // ── Period dropdown (Today / This Week / This Month / This Year) ──
  const periodBtn  = document.getElementById('period-btn');
  const periodMenu = document.getElementById('period-menu');

  periodBtn.addEventListener('click', e => {
    e.stopPropagation();
    periodMenu.classList.toggle('hidden');
  });

  document.querySelectorAll('#period-menu li').forEach(item => {
    item.addEventListener('click', function(e) {
      e.stopPropagation();
      activePeriod = this.getAttribute('data-period');
      document.getElementById('period-btn-label').textContent = this.textContent;
      periodMenu.classList.add('hidden');
      renderPeriodCount();
    });
  });

  document.addEventListener('click', e => {
    if (periodMenu && !periodBtn.contains(e.target)) {
      periodMenu.classList.add('hidden');
    }
  });

  updateDashboard();
  setInterval(updateDashboard, 3000);

  const filterBtn  = document.getElementById("filter-btn");
  const filterMenu = document.getElementById("filter-menu");
  const dateRange  = document.getElementById("date-range");

  const elStartDate = document.getElementById("start-date");
  const elEndDate   = document.getElementById("end-date");

  const startMonthGroup = document.getElementById("start-month-group");
  const endMonthGroup   = document.getElementById("end-month-group");
  const elStartMonthSel = document.getElementById("start-month-sel");
  const elStartYearSel  = document.getElementById("start-month-year");
  const elEndMonthSel   = document.getElementById("end-month-sel");
  const elEndYearSel    = document.getElementById("end-month-year");

  const elStartYear = document.getElementById("start-year");
  const elEndYear   = document.getElementById("end-year");

  const applyBtn  = document.getElementById("apply-range");
  let currentType = null;

  const allInputs = [
    elStartDate, elEndDate,
    startMonthGroup, endMonthGroup,
    elStartYear, elEndYear
  ];

  filterBtn.addEventListener("click", e => {
    e.stopPropagation();
    filterMenu.classList.toggle("hidden");
    dateRange.classList.add("hidden");
  });

  document.querySelectorAll("#filter-menu li").forEach(item => {
    item.addEventListener("click", function (e) {
      e.stopPropagation();
      currentType = this.getAttribute("data-type");
      filterMenu.classList.add("hidden");
      allInputs.forEach(el => el.classList.add("hidden"));
      dateRange.classList.remove("hidden");

      if (currentType === "daily" || currentType === "weekly") {
        elStartDate.classList.remove("hidden");
        elEndDate.classList.remove("hidden");
      } else if (currentType === "monthly") {
        startMonthGroup.classList.remove("hidden");
        endMonthGroup.classList.remove("hidden");
      } else if (currentType === "yearly") {
        elStartYear.classList.remove("hidden");
        elEndYear.classList.remove("hidden");
      }

      document.getElementById("chart-title").textContent =
        currentType.charAt(0).toUpperCase() + currentType.slice(1) + " Banana Sorting";
    });
  });

  applyBtn.addEventListener("click", function () {
    if (!currentType) return;
    let startVal, endVal;

    if (currentType === "daily" || currentType === "weekly") {
      startVal = elStartDate.value;
      endVal   = elEndDate.value;
      if (!startVal || !endVal) { alert("Please select both a start and end date."); return; }

    } else if (currentType === "monthly") {
      const sM = parseInt(elStartMonthSel.value);
      const sY = parseInt(elStartYearSel.value);
      const eM = parseInt(elEndMonthSel.value);
      const eY = parseInt(elEndYearSel.value);
      if (!sM || !sY || !eM || !eY) { alert("Please select both a start and end month/year."); return; }
      if (sY > eY || (sY === eY && sM > eM)) { alert("Start month/year must be before or equal to the end month/year."); return; }
      startVal = { month: sM, year: sY };
      endVal   = { month: eM, year: eY };

    } else if (currentType === "yearly") {
      startVal = elStartYear.value;
      endVal   = elEndYear.value;
      if (!startVal || !endVal) { alert("Please enter both a start and end year."); return; }
    }

    activeFilter = { type: currentType, startVal, endVal };
    applyFilterToChart(currentType, startVal, endVal);
    dateRange.classList.add("hidden");
  });

  document.addEventListener("click", e => {
    const filter = document.querySelector(".filter");
    if (!filter.contains(e.target)) {
      filterMenu.classList.add("hidden");
      dateRange.classList.add("hidden");
    }
  });
});