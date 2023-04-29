/* eslint-disable react/prop-types */
import { memo, useEffect, useRef } from 'react'
// eslint-disable-next-line react/display-name
const CanvasMatrix = memo(({ matrix, width, height, rows }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        const cellSize = Math.min(width / rows, height / rows);

        for (let i = 0; i < matrix.length; i++) {
            const x = i % rows; // column index
            const y = Math.floor(i / rows); // row index
            context.fillStyle = matrix[i] ? 'black' : 'white';
            context.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }, [matrix, width, height, rows]);

    return <canvas ref={canvasRef} width={width} height={height} />;
});

export default CanvasMatrix;