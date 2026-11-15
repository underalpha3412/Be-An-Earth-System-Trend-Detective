export type ParameterKey='PRECTOTCORR'|'QV2M'|'T2M_MAX'|'T2M_MIN'|'TSOIL1'|'WS10M';
export type Parameter={label:string;unit:string;color:string;ann:Record<string,number|null>;monthly:Record<string,number[]>};
export type HistoricalData={years:number[];months:string[];location:{name:string;latitude:number;longitude:number};parameters:Record<ParameterKey,Parameter>};
export type ForecastHorizon={temperature_c:number;precipitation_mm:number;humidity_pct:number;wind_ms:number;rain_probability:number;valid_time_ist:string;dominant_engine:string;temp_uncertainty_90:[number|null,number|null]};
export type ForecastResult={available:boolean;horizons:Record<string,ForecastHorizon>;metadata:Record<string,unknown>;error?:string|null};
