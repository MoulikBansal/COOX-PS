# COOX Geo-Blocking Strategy using Spatial Analytics

## Overview
Built a **data-driven geo-blocking strategy** for **COOX** by analyzing **500+ historically refunded (non-serviceable) bookings**.

The objective was to identify geographic hotspots where COOX repeatedly failed to serve customers and proactively block those areas on the platform to reduce refunds, improve operational efficiency, and enhance customer experience.

This project transforms raw booking latitude/longitude data into actionable business decisions using clustering, geospatial intelligence, and interactive analytics.

---

## Problem Statement
COOX was receiving bookings from areas that were repeatedly marked as **non-serviceable**, resulting in:

- Customer refunds  
- Poor customer experience  
- Operational inefficiencies  
- Preventable revenue loss  

Instead of reacting after failed bookings, the goal was to build a system that could identify such zones in advance and recommend geo-blocking proactively.

---

## Solution Approach

### 1. Spatial Hotspot Detection
Used **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** with:

- **Haversine distance metric** for geo-coordinates  
- **5 km clustering radius**  
- Density-based grouping to detect natural hotspots from refunded bookings  

This allowed discovery of problematic zones without predefining the number of clusters.

---

### 2. Reverse Geocoding for Actionability
Used **Nominatim / OpenStreetMap** reverse geocoding to convert cluster centroids into:

- Pin codes  
- Cities  
- States  

This converted raw coordinates into platform-ready blocking recommendations.

---

### 3. Interactive Visual Analytics

#### Geo Heatmap (`coox_map.html`)
Features:

- Booking density heatmap across India  
- Cluster markers  
- Multiple tile layers:
  - Light mode  
  - Dark mode  
  - Satellite view  

#### Analytics Dashboard (`coox_report.html`)
Built using **Chart.js** with:

- City-wise refunded booking distribution  
- State-wise breakdown  
- Cluster size distribution  
- Geo-blocking recommendations  
- Executive summary insights  

---

## Key Outputs

| File | Description |
|------|-------------|
| `coox_map.html` | Interactive map showing booking density and hotspot clusters |
| `coox_report.html` | Analytics dashboard with charts and recommendations |
| `coox_clusters.csv` | Ready-to-implement geo-block pin code list |

---

## Results & Business Impact

- Identified that a **small set of outskirts pin codes** accounted for a disproportionate share of failed bookings.
- Generated a **~70 pin code blocking list** for immediate implementation.
- Reduced avoidable refund costs.
- Improved customer experience by preventing bookings in unreachable areas.
- Enabled data-driven operational decision-making.

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- DBSCAN  
- Geopy  
- Nominatim / OpenStreetMap  
- Folium  
- Chart.js  
- HTML / CSS / JavaScript  

---

## Core Insight

A relatively small number of recurring non-serviceable areas were driving a significant portion of refunds. Proactively blocking these zones can meaningfully improve fulfillment success and platform efficiency.

---

## Future Enhancements

- Real-time serviceability checks during checkout  
- Dynamic unblocking based on vendor expansion  
- Travel-time based serviceability instead of radius-only logic  
- Automated monthly hotspot refresh pipeline  
- Integration with internal logistics systems  

---
