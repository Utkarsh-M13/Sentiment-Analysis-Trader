import SharpeGauge from "./SharpeGuage"

const SharpeRatio = () => {
  return (
    <div className="w-40 h-40 rounded-lg purple-shadow">
      <SharpeGauge value={2.5}></SharpeGauge>
    </div>
  )
}

export default SharpeRatio