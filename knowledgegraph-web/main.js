let rawData = [];
let graphData = { nodes: [], links: [] };
let chartInstance = null;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        // 1. 获取数据
        const response = await fetch('./data/standard_cut.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        rawData = await response.json();

        // 2. 处理数据
        graphData = processKnowledgeGraph(rawData);

        // 3. 初始化图表
        chartInstance = initChart('graphChart', graphData, handleNodeClick);

        // 4. 初始渲染论文列表
        renderPaperList(rawData);

        // 5. 隐藏加载动画
        document.getElementById('loader').style.display = 'none';

        // 6. 绑定搜索事件
        document.getElementById('searchInput').addEventListener('input', (e) => {
            const keyword = e.target.value.toLowerCase().trim();
            filterPapers(keyword);
        });

        // 7. 绑定关闭按钮点击事件
        const closeBtn = document.getElementById('closeDetailsBtn');
        if (closeBtn) {
            closeBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // 防止冒泡
                closeDetailsPanel();
            });
        }

        // 8. 绑定图表空白区域点击事件 (点击空白处撤销)
        chartInstance.getZr().on('click', (params) => {
            if (!params.target) {
                closeDetailsPanel();
            }
        });

    } catch (error) {
        console.error('Error loading data:', error);
        const loader = document.getElementById('loader');
        loader.innerHTML = '<p style="color:red; padding:20px;">数据加载失败。<br>请检查：<br>1. 是否存在 ./data/standard_cut_2.json 文件。<br>2. 浏览器控制台的具体报错信息。</p>';
    }
});

/**
 * 关闭详情面板并重置视图
 */
function closeDetailsPanel() {
    document.getElementById('detailsPanel').style.display = 'none';
    // 重置高亮状态，恢复默认视图
    highlightNodes([], []);
    // 恢复显示所有论文列表（如果之前因为点击实体节点筛选了列表）
    renderPaperList(rawData);
}

/**
 * 核心逻辑：处理知识图谱数据
 */
function processKnowledgeGraph(data) {
    const nodes = new Map(); 
    const links = [];

    data.forEach(paper => {
        // --- Step 1: 添加论文节点 ---
        if (!nodes.has(paper.paper_id)) {
            nodes.set(paper.paper_id, {
                id: paper.paper_id,
                name: paper.title,
                category: '论文',
                symbolSize: 40,
                paperData: paper 
            });
        }

        // --- Step 2: 构建实体类型映射 ---
        const entityTypes = {};

        const mapEntity = (arr, type) => {
            if (arr && Array.isArray(arr)) {
                arr.forEach(item => {
                    let name;
                    if (typeof item === 'string') {
                        name = item;
                    } else {
                        if (type === '创新点') {
                            name = item.description;
                        } else {
                            name = item.name || item.description;
                        }
                    }
                    
                    if (name) {
                        entityTypes[name] = type;
                    }
                });
            }
        };

        mapEntity(paper.methods, '方法');
        mapEntity(paper.tasks, '任务');
        mapEntity(paper.imaging_modalities, '模态');
        mapEntity(paper.datasets, '数据集');
        mapEntity(paper.anatomical_structures, '解剖结构');
        mapEntity(paper.metrics, '指标'); 
        mapEntity(paper.innovations, '创新点'); 

        // --- Step 3: 处理关系 ---
        if (paper.relations && Array.isArray(paper.relations)) {
            paper.relations.forEach(rel => {
                const sourceId = rel.from;
                const targetId = rel.to;

                ensureNode(sourceId, entityTypes, paper);
                ensureNode(targetId, entityTypes, paper);

                links.push({
                    source: sourceId,
                    target: targetId,
                    name: rel.type,
                    lineStyle: { curveness: 0.2, opacity: 0.6 }
                });
            });
        }
    });

    return {
        nodes: Array.from(nodes.values()),
        links: links
    };

    function ensureNode(nodeId, typeMap, contextPaper) {
        if (!nodes.has(nodeId)) {
            let category = typeMap[nodeId];
            
            if (!category) {
                category = inferCategory(nodeId);
            }

            nodes.set(nodeId, {
                id: nodeId,
                name: nodeId, 
                category: category,
                symbolSize: 15
            });
        }
    }
}

function inferCategory(name) {
    if (!name) return '未知';
    const n = name.toLowerCase();
    
    if (n.includes('mri') || n.includes('ct') || n.includes('x-ray') || n.includes('x光') || 
        n.includes('ultrasound') || n.includes('超声') || n.includes('pet') || 
        n.includes('spect') || n.includes('oct') || n.includes('eeg') || n.includes('光')) {
        return '模态';
    }
    if (n.includes('segmentation') || n.includes('分割') || 
        n.includes('classification') || n.includes('分类') || 
        n.includes('detection') || n.includes('检测') || 
        n.includes('registration') || n.includes('配准') || 
        n.includes('reconstruction') || n.includes('重建') ||
        n.includes('定位')) {
        return '任务';
    }
    if (n.includes('accuracy') || n.includes('准确率') || 
        n.includes('dice') || n.includes('iou') || n.includes('sensitivity') || n.includes('敏感度') ||
        n.includes('precision') || n.includes('精度') || 
        n.includes('f1') || n.includes('auc') || n.includes('psnr') || n.includes('ssim')) {
        return '指标';
    }
    if (n.includes('dataset') || n.includes('数据集') || n.includes('challenge') || 
        n.includes('adni') || n.includes('mimic') || n.includes('chest')) {
        return '数据集';
    }
    if (n.includes('innov') || n.includes('创新') || n.includes('proposed') || n.includes('提出')) {
        return '创新点';
    }
    
    return '未知';
}

function filterPapers(keyword) {
    if (!keyword) {
        renderPaperList(rawData);
        highlightNodes([], []); 
        return;
    }

    const matchedPapers = rawData.filter(paper => {
        const contentStr = [
            paper.title,
            paper.doi,
            ...(paper.authors || []),
            ...(paper.methods?.map(m => typeof m === 'string' ? m : m.name) || []),
            ...(paper.tasks || []),
            ...(paper.imaging_modalities || []),
            ...(paper.innovations?.map(i => i.description) || [])
        ].join(' ').toLowerCase();
        return contentStr.includes(keyword);
    });

    renderPaperList(matchedPapers);

    const highlightIds = new Set();
    const relatedLinks = [];

    matchedPapers.forEach(paper => {
        highlightIds.add(paper.paper_id);
        
        if (paper.relations) {
            paper.relations.forEach(rel => {
                if (rel.from.toLowerCase().includes(keyword) || rel.to.toLowerCase().includes(keyword)) {
                    highlightIds.add(rel.from);
                    highlightIds.add(rel.to);
                }
                
                relatedLinks.push({
                    source: paper.paper_id,
                    target: rel.to
                });
            });
        }
    });

    highlightNodes(Array.from(highlightIds), relatedLinks);
}

/**
 * 高亮节点逻辑
 * 【优化】移除了 symbolSize 的动态修改，防止触发力导向图的重新计算，从而解决交互卡顿问题
 */
function highlightNodes(activeIds, activeLinks) {
    // 如果没有传入高亮ID，说明是“撤销”操作，恢复原始数据
    if (!activeIds || activeIds.length === 0) {
        chartInstance.setOption({
            series: [{
                data: graphData.nodes, // 恢复所有节点
                links: graphData.links,
                lineStyle: {
                    color: 'source',
                    curveness: 0.3,
                    width: 1,
                    opacity: 0.5
                }
            }]
        });
        return;
    }

    // 执行高亮逻辑
    const data = graphData.nodes.map(node => {
        const isActive = activeIds.includes(node.id);
        return {
            ...node,
            itemStyle: {
                opacity: isActive ? 1 : 0.1,
                color: isActive ? undefined : '#ccc'
            },
            // 【注意】这里移除了 symbolSize 的赋值。
            // 在力导向图中动态改变节点大小会导致重新计算物理布局，产生严重卡顿。
            // 我们仅通过透明度和颜色来区分高亮状态。
        };
    });

    const links = graphData.links.map(link => {
        const isSourceActive = activeIds.includes(link.source);
        const isTargetActive = activeIds.includes(link.target);
        
        return {
            ...link,
            lineStyle: {
                opacity: (isSourceActive && isTargetActive) ? 1 : 0.05,
                width: (isSourceActive && isTargetActive) ? 2 : 1
            }
        };
    });

    chartInstance.setOption({
        series: [{
            data: data,
            links: links
        }]
    });
}

function handleNodeClick(nodeData) {
    const detailsPanel = document.getElementById('detailsPanel');
    
    // 显示面板
    detailsPanel.style.display = 'block';

    if (nodeData.category === '论文' && nodeData.paperData) {
        const p = nodeData.paperData;
        document.getElementById('detailTitle').innerText = p.title;
        document.getElementById('detailAuthors').innerText = (p.authors || []).join(', ');
        document.getElementById('detailDoi').innerText = p.doi || '-';
        document.getElementById('detailDoi').href = p.doi && p.doi.startsWith('http') ? p.doi : `https://doi.org/${p.doi}`;
        document.getElementById('detailCategory').innerText = p.category || '-';

        const innList = document.getElementById('detailInnovations');
        innList.innerHTML = '';
        if (p.innovations && p.innovations.length > 0) {
            p.innovations.forEach(inn => {
                const li = document.createElement('li');
                li.innerText = inn.description;
                innList.appendChild(li);
            });
        } else {
            innList.innerHTML = '<li>无</li>';
        }

        const methodsSpan = document.getElementById('detailMethods');
        methodsSpan.innerText = (p.methods || []).map(m => m.name).join(', ') || '-';

        const metricsSpan = document.getElementById('detailMetrics');
        metricsSpan.innerText = (p.metrics || []).map(m => `${m.name}: ${m.value}`).join(', ') || '-';

        const neighbors = new Set();
        graphData.links.forEach(link => {
            if (link.source === nodeData.id) neighbors.add(link.target);
            if (link.target === nodeData.id) neighbors.add(link.source);
        });
        highlightNodes(Array.from(neighbors), []);

    } else {
        // 点击实体节点
        document.getElementById('detailTitle').innerText = nodeData.name;
        document.getElementById('detailAuthors').innerText = `类型: ${nodeData.category}`;
        document.getElementById('detailDoi').innerText = '';
        document.getElementById('detailCategory').innerText = '-';
        document.getElementById('detailInnovations').innerHTML = '';
        document.getElementById('detailMethods').innerText = '-';
        document.getElementById('detailMetrics').innerText = '-';

        const matchedPapers = rawData.filter(p => {
            const checkList = (arr) => arr ? arr.some(item => {
                if (typeof item === 'string') return item === nodeData.name;
                if (item.description) return item.description === nodeData.name;
                return item.name === nodeData.name;
            }) : false;
            
            return checkList(p.methods) || 
                   checkList(p.tasks) || 
                   checkList(p.imaging_modalities) || 
                   checkList(p.datasets) || 
                   checkList(p.innovations);
        });
        
        renderPaperList(matchedPapers);
        
        const highlightIds = new Set([nodeData.id, ...matchedPapers.map(p => p.paper_id)]);
        highlightNodes(Array.from(highlightIds), []);
    }
}

function renderPaperList(papers) {
    const listContainer = document.getElementById('paperList');
    listContainer.innerHTML = '';

    if (!papers || papers.length === 0) {
        listContainer.innerHTML = '<div style="padding:15px; color:#999;">无匹配论文</div>';
        return;
    }

    papers.slice(0, 50).forEach(paper => {
        const item = document.createElement('div');
        item.className = 'paper-item';
        item.innerHTML = `
            <div class="paper-title">${paper.title || '无标题'}</div>
            <div class="paper-authors">${(paper.authors || []).join(', ') || '未知作者'}</div>
        `;
        item.onclick = () => {
            const nodeData = graphData.nodes.find(n => n.id === paper.paper_id);
            if (nodeData) {
                handleNodeClick(nodeData);
            }
        };
        listContainer.appendChild(item);
    });
}







