function initChart(domId, data, clickCallback) {
    const myChart = echarts.init(document.getElementById(domId));

    const categories = [
        { name: '论文', itemStyle: { color: '#5470c6' } },
        { name: '方法', itemStyle: { color: '#91cc75' } },
        { name: '任务', itemStyle: { color: '#fac858' } },
        { name: '模态', itemStyle: { color: '#ee6666' } },
        { name: '数据集', itemStyle: { color: '#73c0de' } },
        { name: '解剖结构', itemStyle: { color: '#2f4554' } },
        { name: '创新点', itemStyle: { color: '#3ba272' } },
        { name: '指标', itemStyle: { color: '#fc8452' } },
        { name: '未知', itemStyle: { color: '#999999' } }
    ];

    const option = {
        toolbox: {
            show: false
        },
        
        tooltip: {
            trigger: 'item',
            formatter: function (params) {
                if (params.dataType === 'edge') {
                    return `${params.data.source} --[${params.data.name}]--> ${params.data.target}`;
                }
                const node = params.data;
                let details = `<b>${node.name}</b><br/>类型: ${node.category}`;
                if (node.category === '论文' && node.paperData) {
                    details += `<br/>DOI: ${node.paperData.doi}`;
                    details += `<br/>作者: ${node.paperData.authors ? node.paperData.authors.slice(0, 3).join(', ') + '...' : ''}`;
                }
                return details;
            }
        },
        legend: [{
            data: categories.map(a => a.name),
            top: 10,
            left: 10,
            textStyle: { fontSize: 10 }
        }],
        series: [
            {
                type: 'graph',
                layout: 'force',
                
                // 【优化1】保持关闭布局动画，瞬间完成，无加载等待
                layoutAnimation: false,

                data: data.nodes.map(node => {
                    const catObj = categories.find(c => c.name === node.category) || categories[categories.length - 1];
                    return {
                        id: node.id,
                        name: node.name,
                        category: node.category,
                        symbolSize: node.symbolSize || 10,
                        itemStyle: { color: catObj.itemStyle.color },
                        paperData: node.paperData
                    };
                }),
                links: data.links,
                categories: categories,
                roam: true,
                
                // 【修改】将 show 改回 true，恢复显示默认标签
                label: {
                    show: true,
                    position: 'right',
                    formatter: '{b}',
                    fontSize: 10
                },
                
                emphasis: {
                    focus: 'adjacency',
                    lineStyle: { width: 3 }
                    // 此时 emphasis 里的 label 配置可以省略，或者保留用于强化显示
                },

                lineStyle: {
                    color: 'source',
                    curveness: 0.3,
                    width: 1,
                    opacity: 0.5
                },
                
                // 【优化4】保持优化后的力导向参数，收敛更快
                force: {
                    repulsion: 1000,
                    edgeLength: 80,
                    gravity: 0.2,
                    friction: 0.8
                }
            }
        ]
    };

    myChart.setOption(option);
    
    myChart.on('click', function (params) {
        if (params.dataType === 'node') {
            clickCallback(params.data);
        }
    });

    window.addEventListener('resize', () => {
        myChart.resize();
    });

    return myChart;
}





