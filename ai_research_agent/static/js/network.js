const NETWORK_API = '/api/network';
let simulation, svg, width, height, selectedNode = null;
let nodeTypeFilters = new Set(['topic','source_type','venue','huggingface','organization','candidate']);
let minEdgeWeight = 1;
const typeColors = {topic:'#6366f1',source_type:'#3b82f6',venue:'#10b981',huggingface:'#a855f7',organization:'#f97316',candidate:'#ef4444'};

async function loadNetwork() {
    const res = await fetch(NETWORK_API);
    if (!res.ok) {
        document.getElementById('stat-nodes').textContent = 'Error';
        document.getElementById('stat-edges').textContent = 'Error';
        return;
    }
    const data = await res.json();
    document.getElementById('stat-nodes').textContent = data.nodes.length;
    document.getElementById('stat-edges').textContent = data.edges.length;
    const types = new Set(data.nodes.map(n=>n.type));
    document.getElementById('stat-types').textContent = types.size;
    const runIds = new Set();
    data.edges.forEach(e=> (e.run_ids||[]).forEach(id=>runIds.add(id)));
    document.getElementById('stat-runs').textContent = runIds.size;

    // Show empty state if no data
    if (data.nodes.length === 0) {
        const svg = d3.select('#network-svg');
        svg.selectAll('*').remove();
        svg.append('text')
            .attr('x', '50%')
            .attr('y', '50%')
            .attr('text-anchor', 'middle')
            .attr('fill', '#6b7280')
            .attr('font-size', '16px')
            .text('No network data yet. Run the agent first to generate run artifacts.');
        return;
    }

    initNetwork(data.nodes, data.edges);
}

function initNetwork(nodeData, edgeData) {
    const container = document.getElementById('network-svg');
    width = Math.max(container.clientWidth, 600);
    height = Math.max(container.clientHeight, 400);
    
    // Ensure SVG has explicit dimensions
    svg = d3.select('#network-svg')
        .attr('width', width)
        .attr('height', height)
        .style('background', '#fafafa');

    const filteredEdges = edgeData.filter(e => e.weight >= minEdgeWeight && nodeTypeFilters.has(e.source.split(':')[0]) && nodeTypeFilters.has(e.target.split(':')[0]));
    
    // Show ALL nodes matching type filters, not just connected ones
    const filteredNodes = nodeData.filter(n => nodeTypeFilters.has(n.type));
    
    // Build adjacency for connected-node highlighting
    const connectedIds = new Set();
    filteredEdges.forEach(e => { connectedIds.add(e.source); connectedIds.add(e.target); });

    const link = svg.append('g').attr('stroke','#999').attr('stroke-opacity',0.6).selectAll('line').data(filteredEdges).join('line').attr('stroke-width', d => Math.sqrt(d.weight)*1.5).attr('stroke','#cbd5e1');
    const node = svg.append('g').attr('stroke','#fff').attr('stroke-width',1.5).selectAll('circle').data(filteredNodes).join('circle').attr('r', d => selectedNode && d.id===selectedNode.id ? 10 : d.type==='topic'?6 : 5).attr('fill', d => typeColors[d.type]||'#94a3b8').attr('stroke', d => connectedIds.has(d.id) ? '#fff' : '#e2e8f0').attr('stroke-width', d => connectedIds.has(d.id) ? 2 : 1).style('cursor','pointer').on('click', (event,d)=>{ selectedNode=d; updateTooltip(d); restart(filteredNodes, filteredEdges); }).on('mouseover', (event,d)=>{ const t=document.getElementById('tooltip'); document.getElementById('tooltip-name').textContent=d.label; document.getElementById('tooltip-type').textContent=d.type; document.getElementById('tooltip-details').innerHTML=`<div>Run count: <strong>${d.run_count}</strong></div><div>Sources: <strong>${d.sources?.length||0}</strong></div><div>Type: <strong>${d.type}</strong></div>`; t.classList.remove('hidden'); t.style.left=(event.pageX+10)+'px'; t.style.top=(event.pageY+10)+'px'; }).on('mouseout', ()=>{ document.getElementById('tooltip').classList.add('hidden'); });
    const label = svg.append('g').attr('pointer-events','none').attr('font-family','Inter,sans-serif').attr('font-size',10).selectAll('text').data(filteredNodes).join('text').text(d=>d.label).attr('fill','#334155').attr('text-anchor','middle').attr('dy', d=> d.type==='topic'?'-1.2em':'1.2em').style('opacity', d => connectedIds.has(d.id) ? 0.9 : 0.4);
    
    simulation = d3.forceSimulation(filteredNodes)
        .force('link', d3.forceLink(filteredEdges).id(d=>d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width/2, height/2))
        .force('collision', d3.forceCollide().radius(d => selectedNode && d.id===selectedNode.id ? 15 : d.type==='topic'?10:7))
        .on('tick', ()=>{ 
            link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
            node.attr('cx',d=>d.x).attr('cy',d=>d.y);
            label.attr('x',d=>d.x).attr('y',d=>d.y); 
        });
}

function restart(nodeData, edgeData) {
    width = Math.max(document.getElementById('network-svg').clientWidth, 600);
    height = Math.max(document.getElementById('network-svg').clientHeight, 400);
    svg.attr('width',width).attr('height',height);
    const filteredEdges = edgeData.filter(e => e.weight >= minEdgeWeight && nodeTypeFilters.has(e.source.split(':')[0]) && nodeTypeFilters.has(e.target.split(':')[0]));
    
    // Show ALL nodes matching type filters
    const filteredNodes = nodeData.filter(n => nodeTypeFilters.has(n.type));
    
    const connectedIds = new Set();
    filteredEdges.forEach(e => { connectedIds.add(e.source); connectedIds.add(e.target); });

    const link = svg.selectAll('line').data(filteredEdges, d=>d.source.id+'-'+d.target.id);
    link.exit().remove(); link.enter().append('line').attr('stroke','#cbd5e1').merge(link).attr('stroke-width', d=>Math.sqrt(d.weight)*1.5);
    const node = svg.selectAll('circle').data(filteredNodes, d=>d.id);
    node.exit().remove(); node.enter().append('circle').attr('fill',d=>typeColors[d.type]||'#94a3b8').attr('stroke',d=>connectedIds.has(d.id)?'#fff':'#e2e8f0').attr('stroke-width',d=>connectedIds.has(d.id)?2:1).style('cursor','pointer').on('click',(event,d)=>{selectedNode=d;updateTooltip(d);restart(filteredNodes,filteredEdges);}).merge(node).attr('r', d=> selectedNode && d.id===selectedNode.id ? 10 : d.type==='topic'?6:5);
    const label = svg.selectAll('text').data(filteredNodes, d=>d.id);
    label.exit().remove(); label.enter().append('text').attr('font-family','Inter,sans-serif').attr('font-size',10).attr('fill','#334155').attr('text-anchor','middle').attr('dy',d=>d.type==='topic'?'-1.2em':'1.2em').style('opacity',d=>connectedIds.has(d.id)?0.9:0.4).merge(label).text(d=>d.label).attr('x',d=>d.x).attr('y',d=>d.y);
    simulation.nodes(filteredNodes); simulation.force('link').links(filteredEdges); simulation.alpha(1).restart();
}

function updateTooltip(d) {
    document.getElementById('tooltip-name').textContent=d.label;
    document.getElementById('tooltip-type').textContent=d.type;
    document.getElementById('tooltip-details').innerHTML=`<div>Run count: <strong>${d.run_count}</strong></div><div>Sources: <strong>${d.sources?.length||0}</strong></div><div>Type: <strong>${d.type}</strong></div>`;
    document.getElementById('tooltip').classList.remove('hidden');
}

document.querySelectorAll('.node-type-filter').forEach(cb=>{ cb.addEventListener('change',(e)=>{ e.target.checked?nodeTypeFilters.add(e.target.dataset.type):nodeTypeFilters.delete(e.target.dataset.type); loadNetwork(); }); });
document.getElementById('edge-weight-filter').addEventListener('input',(e)=>{ minEdgeWeight=parseInt(e.target.value); loadNetwork(); });
document.getElementById('reset-network').addEventListener('click',()=>{ selectedNode=null; loadNetwork(); });
document.getElementById('zoom-in').addEventListener('click',()=>{ if(simulation) simulation.alpha(0.3).restart(); });
document.getElementById('zoom-out').addEventListener('click',()=>{ if(simulation) simulation.alpha(0.1).restart(); });
document.addEventListener('mousemove',(e)=>{ const t=document.getElementById('tooltip'); if(!t.classList.contains('hidden')){ t.style.left=(e.pageX+10)+'px'; t.style.top=(e.pageY+10)+'px'; } });

loadNetwork();